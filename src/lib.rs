#![doc = include_str!("../README.md")]
use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView3, Ix3, OwnedRepr};
use ort::{GraphOptimizationLevel, Session};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::ops::Range;
use std::path::Path;
use std::time::Duration;

/// Parameters used to configure a vad session. These will determine the sensitivity and switching
/// speed of detection.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub pre_speech_pad: Duration,
    pub redemption_time: Duration,
    pub sample_rate: usize,
    pub min_speech_time: Duration,

    #[cfg_attr(feature = "serde", serde(default))]
    pub backend: VadBackend,
}

/// A VAD session create one of these for each audio stream you want to detect voice activity on
/// and feed the audio into it.
#[derive(Debug)]
pub struct VadSession {
    config: VadConfig,
    model: VadModel,
    h_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    c_tensor: ArrayBase<OwnedRepr<f32>, Ix3>,
    sample_rate: usize,
    state: VadState,
    session_audio: Vec<f32>,
    processed_samples: usize,
    silent_samples: usize,
    speech_start: Option<usize>,
    speech_end: Option<usize>,
}

/// Current state of the VAD (speaking or silent)
#[derive(Clone, Debug)]
enum VadState {
    Speech {
        start_ms: usize,
        redemption_passed: bool,
        speech_time: Duration,
    },
    Silence,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VadBackend {
    Ort,
    #[default]
    RTen,
}

enum VadModel {
    Ort(Session),
    RTen(Model),
}

impl std::fmt::Debug for VadModel {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            fmt,
            "{}",
            match self {
                VadModel::Ort(_) => "VadModel::Ort",
                VadModel::RTen(_) => "VadModel::RTen",
            }
        )
    }
}

impl VadModel {
    fn run(
        &self,
        h_tensor: ArrayView3<f32>,
        c_tensor: ArrayView3<f32>,
        sample_rate: usize,
        input: &[f32],
    ) -> Result<(Array3<f32>, Array3<f32>, f32)> {
        match self {
            VadModel::Ort(session) => {
                let samples = input.len();
                let audio_tensor = Array2::from_shape_vec((1, samples), input.to_vec())?;
                let sample_rate_tensor = Array1::from_vec(vec![sample_rate as i64]);
                let mut result = session.run(ort::inputs![
                    audio_tensor.view(),
                    sample_rate_tensor.view(),
                    h_tensor,
                    c_tensor,
                ]?)?;

                // Update internal state tensors.
                let h_tensor = result
                    .get("hn")
                    .unwrap()
                    .try_extract_tensor::<f32>()?
                    .to_owned()
                    .into_shape((2, 1, 64))
                    .context("Shape mismatch for h_tensor")?;

                let c_tensor = result
                    .get("cn")
                    .unwrap()
                    .try_extract_tensor::<f32>()?
                    .to_owned()
                    .into_shape((2, 1, 64))
                    .context("Shape mismatch for h_tensor")?;

                let prob = result
                    .remove("output")
                    .unwrap()
                    .try_extract_tensor::<f32>()?
                    .into_shape((1, 1))?
                    .first()
                    .copied()
                    .ok_or(anyhow!("empty prob output"))?;

                Ok((h_tensor, c_tensor, prob))
            }
            #[allow(unreachable_code)]
            VadModel::RTen(model) => {
                let input_id = model.node_id("input")?;
                let h_id = model.node_id("h")?;
                let c_id = model.node_id("c")?;
                let sr_id = model.node_id("sr")?;
                let hn_id = model.node_id("hn")?;
                let cn_id = model.node_id("cn")?;
                let prob_id = model.node_id("output")?;

                let input_tensor = NdTensorView::from_data([1, input.len()], input);
                let h_tensor_rten = h_tensor
                    .as_slice()
                    .map(|x| NdTensorView::from_data(h_tensor.dim().into(), x))
                    .ok_or(anyhow!("h input not contiguous"))?;
                let c_tensor_rten = c_tensor
                    .as_slice()
                    .map(|x| NdTensorView::from_data(c_tensor.dim().into(), x))
                    .ok_or(anyhow!("c input not contiguous"))?;
                let sr_tensor_rten = NdTensor::from_scalar(sample_rate as i32);

                let [hn, cn, prob] = model.run_n(
                    [
                        (input_id, input_tensor.into()),
                        (h_id, h_tensor_rten.into()),
                        (c_id, c_tensor_rten.into()),
                        (sr_id, sr_tensor_rten.into()),
                    ]
                    .into(),
                    [hn_id, cn_id, prob_id],
                    None,
                )?;

                let h_tensor: NdTensor<f32, 3> = hn.try_into()?;
                let h_tensor = Array3::from_shape_vec(h_tensor.shape(), h_tensor.into_data())?;

                let c_tensor: NdTensor<f32, 3> = cn.try_into()?;
                let c_tensor = Array3::from_shape_vec(c_tensor.shape(), c_tensor.into_data())?;

                let prob: NdTensor<f32, 2> = prob.try_into()?;
                let prob = prob
                    .get([0, 0])
                    .copied()
                    .ok_or(anyhow!("empty prob output"))?;

                Ok((h_tensor, c_tensor, prob))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VadTransition {
    SpeechStart {
        /// When the speech started, in milliseconds since the start of the VAD session.
        timestamp_ms: usize,
    },
    SpeechEnd {
        /// When the speech ended, in milliseconds since the start of the VAD session.
        timestamp_ms: usize,
    },
}

impl VadSession {
    /// Create a new VAD session loading an onnx file from the specified path and using the
    /// provided config.
    pub fn new_from_path(file: impl AsRef<Path>, config: VadConfig) -> Result<Self> {
        let bytes = std::fs::read(file.as_ref())
            .with_context(|| format!("Couldn't read onnx file: {}", file.as_ref().display()))?;
        Self::new_from_bytes(&bytes, config)
    }

    /// Create a new VAD session loading an onnx file from memory and using the provided config.
    pub fn new_from_bytes(model_bytes: &[u8], config: VadConfig) -> Result<Self> {
        if ![8000_usize, 16000].contains(&config.sample_rate) {
            bail!("Unsupported sample rate, use 8000 or 16000!");
        }

        let model = match config.backend {
            VadBackend::Ort => {
                let session = Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(4)?
                    .commit_from_memory(model_bytes)?;
                VadModel::Ort(session)
            }
            VadBackend::RTen => {
                let model = Model::load(model_bytes.to_vec())?;
                VadModel::RTen(model)
            }
        };
        let h_tensor = Array3::<f32>::zeros((2, 1, 64));
        let c_tensor = Array3::<f32>::zeros((2, 1, 64));

        Ok(Self {
            config,
            model,
            h_tensor,
            c_tensor,
            sample_rate: config.sample_rate,
            state: VadState::Silence,
            session_audio: vec![],
            processed_samples: 0,
            silent_samples: 0,
            speech_start: None,
            speech_end: None,
        })
    }

    /// Create a new VAD session using the provided config. The ONNX file has been statically
    /// embedded within the library so this will increase binary size by 1.7M.
    #[cfg(feature = "static-model")]
    pub fn new(config: VadConfig) -> Result<Self> {
        let ort_model_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/silero_vad.onnx"
        ));

        let rten_model_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/silero_vad.rten"
        ));

        let model_bytes = match config.backend {
            VadBackend::Ort => ort_model_bytes,
            VadBackend::RTen => rten_model_bytes,
        };

        Self::new_from_bytes(model_bytes, config)
    }

    /// Pass in some audio to the VAD and return a list of any speech transitions that happened
    /// during the segment.
    pub fn process(&mut self, audio_frame: &[f32]) -> Result<Vec<VadTransition>> {
        const VAD_BUFFER_MS: usize = 30; // TODO This should be configurable
        let vad_segment_length = VAD_BUFFER_MS * self.config.sample_rate / 1000;

        let unprocessed = self.session_audio.len() - self.processed_samples;
        let num_chunks = (unprocessed + audio_frame.len()) / vad_segment_length;

        self.session_audio.extend_from_slice(audio_frame);

        let mut transitions = vec![];

        for i in 0..num_chunks {
            // we might not be getting audio chunks in perfect multiples of 30ms, so let the
            // last frame accommodate the remainder. This adds a bit of non-determinism based on
            // audio size but it does let us more eagerly process audio.
            //
            // processed_samples is updated in process_internal so always points to the index of
            // the next sample to go from.
            let sample_range = if i < num_chunks - 1 {
                self.processed_samples..(self.processed_samples + vad_segment_length)
            } else {
                self.processed_samples..self.session_audio.len()
            };
            let vad_result = self.process_internal(sample_range)?;

            if let Some(vad_ev) = vad_result {
                transitions.push(vad_ev);
            }
        }
        Ok(transitions)
    }

    pub fn forward(&mut self, input: Vec<f32>) -> Result<f32> {
        let (h_tensor, c_tensor, prob) = self.model.run(
            self.h_tensor.view(),
            self.c_tensor.view(),
            self.sample_rate,
            &input,
        )?;

        self.h_tensor = h_tensor;
        self.c_tensor = c_tensor;

        Ok(prob)
    }

    /// Advance the VAD state machine with an audio frame. Keep between 30-96ms in length.
    /// Return indicates if a transition from speech to silence (or silence to speech) occurred.
    ///
    /// Important: don't implement your own endpointing logic.
    /// Instead, when a `SpeechEnd` is returned, you can use the `get_current_speech()` method to retrieve the audio.
    fn process_internal(&mut self, range: Range<usize>) -> Result<Option<VadTransition>> {
        let audio_frame = self.session_audio[range].to_vec();
        let samples = audio_frame.len();

        let prob = self.forward(audio_frame)?;

        let mut vad_change = None;

        if prob < self.config.negative_speech_threshold {
            self.silent_samples += samples;
        } else {
            self.silent_samples = 0;
        }

        let current_silence = self.current_silence_duration();

        match self.state {
            VadState::Silence => {
                if prob > self.config.positive_speech_threshold {
                    self.state = VadState::Speech {
                        start_ms: self
                            .session_time()
                            .saturating_sub(self.config.pre_speech_pad)
                            .as_millis() as usize,
                        redemption_passed: false,
                        speech_time: Duration::ZERO,
                    };
                }
            }
            VadState::Speech {
                start_ms,
                ref mut redemption_passed,
                ref mut speech_time,
            } => {
                *speech_time +=
                    Duration::from_secs_f64(samples as f64 / self.config.sample_rate as f64);
                if !*redemption_passed && *speech_time > self.config.min_speech_time {
                    *redemption_passed = true;
                    // TODO: the pre speech padding should not cross over the previous speech->silence
                    // transition, if there was one
                    vad_change = Some(VadTransition::SpeechStart {
                        timestamp_ms: start_ms,
                    });
                    self.speech_start = Some(start_ms);
                    self.speech_end = None;
                }

                if prob < self.config.negative_speech_threshold {
                    if !*redemption_passed {
                        self.state = VadState::Silence;
                    } else {
                        if current_silence > self.config.redemption_time {
                            if *redemption_passed {
                                let speech_end = (self.processed_samples + samples
                                    - self.silent_samples)
                                    / (self.config.sample_rate / 1000);
                                vad_change = Some(VadTransition::SpeechEnd {
                                    timestamp_ms: speech_end,
                                });
                                self.speech_end = Some(speech_end);
                            }
                            self.state = VadState::Silence
                        }
                    }
                }
            }
        };

        self.processed_samples += samples;

        Ok(vad_change)
    }

    /// Returns whether the vad current believes the audio to contain speech
    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speech {
            redemption_passed, ..
        } if redemption_passed)
    }

    /// Gets a buffer of the most recent active speech frames from the time the speech started to the
    /// end of the speech. Parameters from `VadConfig` have already been applied here so this isn't
    /// derived from the raw VAD inferences but instead after padding and filtering operations have
    /// been applied.
    pub fn get_current_speech(&self) -> &[f32] {
        if let Some(speech_start) = self.speech_start {
            let speech_start = speech_start * (self.config.sample_rate / 1000);
            if let Some(speech_end) = self.speech_end {
                let speech_end = speech_end * (self.config.sample_rate / 1000);
                &self.session_audio[speech_start..speech_end]
            } else {
                &self.session_audio[speech_start..]
            }
        } else {
            &[]
        }
    }

    /// Get how long the current speech is in samples.
    pub fn current_speech_samples(&self) -> usize {
        self.get_current_speech().len()
    }

    /// Returns the duration of the current speech segment. It is possible for this and
    /// `Self::current_silence_duration` to both report >0s at  the same time as this takes into
    /// account the switching and padding parameters of the VAD whereas the silence measure ignores
    /// them instead of just focusing on raw network output.
    pub fn current_speech_duration(&self) -> Duration {
        Duration::from_millis(
            (self.current_speech_samples() / (self.config.sample_rate / 1000)) as u64,
        )
    }

    /// Get the current length of the VAD session.
    pub fn session_time(&self) -> Duration {
        Duration::from_secs_f64(self.processed_samples as f64 / self.config.sample_rate as f64)
    }

    /// Reset the status of the model
    // TODO should this reset the audio buffer as well?
    pub fn reset(&mut self) {
        self.h_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.c_tensor = Array3::<f32>::zeros((2, 1, 64));
        self.speech_start = None;
        self.speech_end = None;
        self.silent_samples = 0;
        self.state = VadState::Silence;
    }

    /// Returns the length of the end silence in number of samples. The VAD may be showing this as
    /// speaking because of redemption frames or other parameters that slow down the speed it can
    /// switch at. But this measure is a raw unprocessed look of how many segments since the last
    /// speech are below the negative speech threshold.
    pub fn current_silence_samples(&self) -> usize {
        self.silent_samples
    }

    /// Returns the duration of the end silence. The VAD may be showing this as speaking because of
    /// redemption frames or other parameters that slow down the speed it can switch at. But this
    /// measure is a raw unprocessed look of how many segments since the last speech are below the
    /// negative speech threshold.
    pub fn current_silence_duration(&self) -> Duration {
        Duration::from_millis((self.silent_samples / (self.config.sample_rate / 1000)) as u64)
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            // https://github.com/ricky0123/vad/blob/ea584aaf66d9162fb19d9bfba607e264452980c3/packages/_common/src/frame-processor.ts#L52
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            pre_speech_pad: Duration::from_millis(600),
            redemption_time: Duration::from_millis(600),
            sample_rate: 16000,
            min_speech_time: Duration::from_millis(90),
            backend: VadBackend::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path(backend: VadBackend) -> &'static str {
        match backend {
            VadBackend::Ort => "models/silero_vad.onnx",
            VadBackend::RTen => "models/silero_vad.rten",
        }
    }

    /// Basic smoke test that the model loads correctly and we haven't committed rubbish to the
    /// repo.
    #[test]
    fn model_loads() {
        let cfg = VadConfig::default();
        let model_path = model_path(cfg.backend);
        let _sesion = VadSession::new(cfg).unwrap();
        let _sesion = VadSession::new_from_path(model_path, VadConfig::default()).unwrap();
    }

    /// Too short tensors result in inference errors which we don't want to unnecessarily bubble up
    /// to the user and instead handle in our buffering implementation. This test will check that a
    /// short inference in the internal inference call bubbles up an error but when using the
    /// public API no error is presented.
    #[test]
    fn short_audio_handling() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();

        let short_audio = vec![0.0; 160];

        session.session_audio = short_audio.clone();
        assert!(session.process_internal(0..160).is_err());
        session.session_audio.clear();
        assert!(session.process(&short_audio).unwrap().is_empty());
    }

    /// Check that a long enough packet of just zeros gets an inference and it doesn't flag as
    /// transitioning to speech
    #[test]
    fn silence_handling() {
        let mut session = VadSession::new(VadConfig::default()).unwrap();
        let silence = vec![0.0; 30 * 16]; // 30ms of silence

        assert!(session.process(&silence).unwrap().is_empty());
        assert_eq!(session.processed_samples, silence.len());
    }

    /// We only allow for 8khz and 16khz audio.
    #[test]
    fn reject_invalid_sample_rate() {
        let mut config = VadConfig::default();
        config.sample_rate = 16000;
        VadSession::new(config.clone()).unwrap();
        config.sample_rate = 8000;
        VadSession::new(config.clone()).unwrap();

        config.sample_rate += 1;
        let model_path = model_path(config.backend);
        assert!(VadSession::new(config.clone()).is_err());
        assert!(VadSession::new_from_path(model_path, config.clone()).is_err());

        let bytes = std::fs::read(model_path).unwrap();
        assert!(VadSession::new_from_bytes(&bytes, config).is_err());
    }
}
