use std::error::Error;

use hound::WavReader;
use silero::{VadConfig, VadSession, VadTransition};

/// Analyzes a 16 kHz audio file and prints the probability of audio being
/// speech for each 30ms chunk, followed by a list of transitions between speech
/// and non-speech sections of the file.
///
/// Usage:
///
/// ```text
/// cargo run --example detect_speech tests/audio/sample_1.wav
/// ```
fn main() -> Result<(), Box<dyn Error>> {
    let config = VadConfig::default();
    let mut vad = VadSession::new(config)?;

    let mut args = std::env::args().skip(1);
    let Some(audio_path) = args.next() else {
        return Err("No audio file specified".into());
    };

    println!("VAD configuration:");
    println!(
        "  positive_speech_threshold={}",
        config.positive_speech_threshold
    );
    println!(
        "  negative_speech_threshold={}",
        config.negative_speech_threshold
    );
    println!("  pre_speech_pad={}", config.pre_speech_pad.as_millis());
    println!("  redemption_time={}", config.redemption_time.as_millis());
    println!("  sample_rate={}", config.sample_rate);
    println!("  min_speech_time={}", config.min_speech_time.as_millis());

    let sample_rate = 16_000u32;
    let chunk_ms = 30;
    let chunk_size: usize = sample_rate as usize / chunk_ms;

    let wav_file = WavReader::open(&audio_path)?;
    let actual_sample_rate = wav_file.spec().sample_rate;
    if actual_sample_rate != sample_rate {
        return Err(format!(
            "Audio file has a same rate of {} but this example expects {}",
            actual_sample_rate, sample_rate
        )
        .into());
    }

    let samples: Vec<f32> = wav_file
        .into_samples()
        .map(|x| x.unwrap_or(0i16) as f32 / (i16::MAX as f32))
        .collect();

    let mut transitions = Vec::new();

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let mut full_chunk = chunk.to_vec();
        let pad_samples = chunk_size - full_chunk.len();
        if pad_samples > 0 {
            full_chunk.extend(std::iter::repeat(0.).take(pad_samples));
        }

        // This currently runs the model twice, once to determine the
        // transitions and again to get the probability for the current audio
        // chunk. It would be useful if it could run just once.
        let chunk_transitions = vad.process(&full_chunk)?;
        transitions.extend(chunk_transitions);

        let output = vad.forward(full_chunk)?;
        let prob = output.try_extract_tensor::<f32>()?[[0, 0]];

        let start_ms = i * chunk_ms;
        let sample_rate_ms = sample_rate / 1000;
        let end_ms = start_ms + (chunk.len() / sample_rate_ms as usize);

        let voice_activity = if let Some(state) = transitions.last() {
            match state {
                VadTransition::SpeechStart { .. } => "yes",
                VadTransition::SpeechEnd { .. } => "no",
            }
        } else {
            "unknown"
        };

        println!(
            "time: {start_ms}..{end_ms}ms speech prob: {:.3} voice: {}",
            prob, voice_activity
        );
    }

    for transition in transitions {
        match transition {
            VadTransition::SpeechStart { timestamp_ms } => {
                println!("speech started at {}ms", timestamp_ms);
            }
            VadTransition::SpeechEnd { timestamp_ms } => {
                println!("speech ended at {}ms", timestamp_ms);
            }
        }
    }

    Ok(())
}
