#include "avas_algorithm.h"
#include "sine_500hz.h"

// Parameters based on UN R138 requirements
#define BASE_FREQ 500.0f
#define V_REF 5.0f
// UN R138 section 6.2.3.2: average of at least 0.8 % per 1 km/h in the speed range from 5 km/h to 20 km/h
#define SHIFT_PER_KMH 0.01f // 1.0% shift per km/h (satisfies >= 0.8%)

#define NUM_OSCILLATORS 4

static float output_sample_rate = 16000.0f;
static float current_speed = 0.0f;
static float phase_accumulators[NUM_OSCILLATORS] = {0};

static int is_honking = 0;
static float honk_phases[2] = {0.0f, 0.0f};
static const float honk_freqs[2] = {400.0f, 500.0f}; // Classic dual-tone car horn (Major 3rd)

// We use 4 oscillators to create a richer, more complex sound resembling an EV motor.
// Frequencies relative to fundamental:
// 1.0x : Fundamental pitch
// 1.5x : A perfect fifth above, adds a mechanical "whine"
// 2.0x : An octave above, adds brightness
// 0.5x : An octave below, adds a "rumble" or low-frequency weight
static const float freq_multipliers[NUM_OSCILLATORS] = {1.0f, 1.5f, 2.0f, 0.5f};

// Amplitudes for each harmonic (sum should be <= 1.0 to prevent clipping)
static const float osc_amplitudes[NUM_OSCILLATORS] = {0.4f, 0.2f, 0.1f, 0.25f};

void avas_init(uint32_t sample_rate) {
    if (sample_rate > 0) {
        output_sample_rate = (float)sample_rate;
    }
    current_speed = 0.0f;
    for (int i = 0; i < NUM_OSCILLATORS; i++) {
        phase_accumulators[i] = 0.0f;
    }
}

void avas_set_speed(float speed_kmh) {
    if (speed_kmh < 0.0f) {
        speed_kmh = -speed_kmh;
    }
    current_speed = speed_kmh;
}

void avas_set_honk(int honking_state) {
    is_honking = honking_state;
}

static float calculate_frequency(float speed) {
    // According to UN R138 4.5.1 (equation 1):
    // del f = {[(f_speed - f_ref)/(v_test - v_ref)]/f_ref} * 100
    // => f_speed = f_ref * (1 + (del_f / 100) * (v_test - v_ref))
    // We use del_f / 100 = SHIFT_PER_KMH
    // The regulation defines f_ref at v_ref = 5 km/h (lowest target speed).
    
    // For speeds below 5 km/h, the behavior is not strictly defined for frequency shift, 
    // but typically it remains proportional down to 0, or holds at f_ref.
    // We will hold it at f_ref for speeds <= 5 km/h to maintain a constant base tone, 
    // and shift it upwards for speeds > 5 km/h.
    float freq = BASE_FREQ;
    
    if (speed > V_REF) {
        freq = BASE_FREQ * (1.0f + SHIFT_PER_KMH * (speed - V_REF));
    }
    
    return freq;
}

#include <math.h>

static float calculate_amplitude(float speed) {
    // ECE R138 section 6.2.4: "The vehicle may emit a sound when stationary."
    // We choose to NOT emit a sound at 0 km/h (idle) to reduce noise pollution.
    if (speed <= 0.001f) {
        return 0.0f;
    }

    // ECE R138 Table 2 requires specific minimum overall SPLs:
    // - 10 km/h: 50 dB(A)
    // - 20 km/h: 56 dB(A)
    // - Reversing (typically ~6 km/h): 47 dB(A)
    //
    // We can perfectly model this relative volume scaling by interpolating in dB.
    // Let's assume 20 km/h corresponds to our maximum digital amplitude (0 dBFS).
    // A 6 dB difference exists between 10 km/h (50) and 20 km/h (56).
    // We map speed to a target dB level, where 20 km/h = 56, 10 km/h = 50, 0 km/h = 44.
    float target_spl_db = 44.0f + (speed / 20.0f) * 12.0f;
    
    // Cap at 56 dB (max speed for AVAS requirements)
    if (target_spl_db > 56.0f) {
        target_spl_db = 56.0f;
    }

    // Convert relative dB to a linear amplitude multiplier 
    // (where 56 dB is reference 1.0 amplitude)
    float amplitude = powf(10.0f, (target_spl_db - 56.0f) / 20.0f);
    
    return amplitude;
}

// Simple fast pseudo-random number generator for white noise (tire simulation)
static uint32_t prng_state = 12345;
static inline float generate_white_noise() {
    prng_state = (1103515245 * prng_state + 12345) & 0x7fffffff;
    // Returns value between -1.0 and 1.0
    return ((float)prng_state / 1073741824.0f) - 1.0f;
}

void avas_generate_samples(int16_t *buffer, size_t num_samples) {
    float target_freq = calculate_frequency(current_speed);
    float target_amp = calculate_amplitude(current_speed);
    
    // Tire noise increases with speed. At 20 km/h, tire noise amplitude is 0.04 (4% of mix for a subtle, cleaner sound)
    float tire_noise_amp = (current_speed / 20.0f) * 0.04f;

    
    float step_sizes[NUM_OSCILLATORS];
    for (int i = 0; i < NUM_OSCILLATORS; i++) {
        // The wavetable is a 500 Hz sine wave originally sampled at 16000 Hz.
        // We scale the step size based on the requested target frequency AND the output sample rate.
        float target_f = target_freq * freq_multipliers[i];
        step_sizes[i] = (target_f / BASE_FREQ) * (16000.0f / output_sample_rate);
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        float mixed_sample = 0.0f;
        
        for (int o = 0; o < NUM_OSCILLATORS; o++) {
            size_t index0 = (size_t)phase_accumulators[o];
            size_t index1 = index0 + 1;
            
            if (index1 >= sine_500hz_length) {
                index1 = 0;
            }
            
            float frac = phase_accumulators[o] - (float)index0;
            
            // Linear interpolation for smoother playback when pitch shifted
            float val0 = (float)sine_500hz_pcm[index0];
            float val1 = (float)sine_500hz_pcm[index1];
            
            float interpolated = val0 + frac * (val1 - val0);
            
            // Mix in this oscillator
            mixed_sample += interpolated * osc_amplitudes[o];
            
            // Advance phase
            phase_accumulators[o] += step_sizes[o];
            while (phase_accumulators[o] >= (float)sine_500hz_length) {
                phase_accumulators[o] -= (float)sine_500hz_length;
            }
        }
        
        // Add tire noise (scaled by speed)
        float noise = generate_white_noise() * tire_noise_amp;
        
        // Combine oscillators (max ~0.95 amplitude) and tire noise (max ~0.15 amplitude)
        // Since we scaled oscillator amplitudes previously, this is safe to add.
        // We multiply by 32767.0f since our wavetable is already 16-bit PCM scale, 
        // wait, the wavetable is ALREADY +/- 32767. 
        // So the noise needs to be scaled to the 16-bit range as well!
        noise *= 32767.0f;
        
        mixed_sample += noise;
        
        // Apply global amplitude scaling to engine sound
        mixed_sample *= target_amp;

        // If honking, mix in the loud car horn (square waves)
        if (is_honking) {
            float honk_mix = 0.0f;
            for (int h = 0; h < 2; h++) {
                // Generate a square wave: +1 or -1
                float square_val = (honk_phases[h] < (sine_500hz_length / 2.0f)) ? 1.0f : -1.0f;
                honk_mix += square_val * 0.5f; // Mix two tones equally
                
                // Advance honk phase
                float honk_step = (honk_freqs[h] / BASE_FREQ) * (16000.0f / output_sample_rate);
                honk_phases[h] += honk_step;
                if (honk_phases[h] >= (float)sine_500hz_length) {
                    honk_phases[h] -= (float)sine_500hz_length;
                }
            }
        // Horn is loud and aggressive (override or mix heavily with engine)
            mixed_sample += honk_mix * 32767.0f * 0.8f; 
        }

        // --- LUFS Normalization Estimation ---
        // To achieve a target of roughly -24.9 LUFS, the RMS amplitude of the signal
        // must be significantly reduced. For a dense sine/square mix, -24.9 LUFS
        // corresponds to a linear gain of approximately 0.12 (varies slightly by crest factor).
        mixed_sample *= 0.12f;
        
        // Anti-clipping protection
        if (mixed_sample > 32767.0f) mixed_sample = 32767.0f;
        if (mixed_sample < -32768.0f) mixed_sample = -32768.0f;
        
        buffer[i] = (int16_t)mixed_sample;
    }
}
