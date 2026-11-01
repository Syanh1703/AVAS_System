import wave
import struct
import re

def main():
    # Read the C header file
    with open('d:\\AVAS\\sine_500hz.h', 'r') as f:
        content = f.read()

    match = re.search(r'const int16_t sine_500hz_pcm\[\] = \{([^}]+)\}', content)
    array_str = match.group(1)
    sine_500hz_pcm = [int(s.strip()) for s in array_str.split(',') if s.strip()]
    sine_500hz_length = len(sine_500hz_pcm)

    # Algorithm parameters from avas_algorithm.c
    BASE_FREQ = 500.0
    V_REF = 5.0
    SHIFT_PER_KMH = 0.01

    NUM_OSCILLATORS = 4
    freq_multipliers = [1.0, 1.5, 2.0, 0.5]
    osc_amplitudes = [0.4, 0.2, 0.1, 0.25]

    def calculate_frequency(speed):
        freq = BASE_FREQ
        if speed > V_REF:
            freq = BASE_FREQ * (1.0 + SHIFT_PER_KMH * (speed - V_REF))
        return freq

    def calculate_amplitude(speed):
        if speed <= 0.001:
            return 0.0
            
        target_spl_db = 44.0 + (speed / 20.0) * 12.0
        if target_spl_db > 56.0:
            target_spl_db = 56.0
            
        amplitude = 10.0 ** ((target_spl_db - 56.0) / 20.0)
        return amplitude

    # Tire noise PRNG equivalent
    prng_state = 12345
    def generate_white_noise():
        nonlocal prng_state
        prng_state = (1103515245 * prng_state + 12345) & 0x7fffffff
        return (prng_state / 1073741824.0) - 1.0

    sample_rate = 16000
    total_seconds = 2  # Simulate accelerating from 0 to 20 km/h over 2 seconds
    total_samples = total_seconds * sample_rate
    
    phase_accumulators = [0.0] * NUM_OSCILLATORS
    
    output_path = 'd:\\AVAS\\avas_simulation_complex.wav'
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(total_samples):
            # Calculate speed for this sample (accelerate from 0 to 20 km/h)
            progress = i / total_samples
            current_speed = progress * 20.0
            
            target_freq = calculate_frequency(current_speed)
            target_amp = calculate_amplitude(current_speed)
            tire_noise_amp = (current_speed / 20.0) * 0.04
            
            step_sizes = [(target_freq * mult) / BASE_FREQ for mult in freq_multipliers]
            
            mixed_sample = 0.0
            
            for o in range(NUM_OSCILLATORS):
                index0 = int(phase_accumulators[o])
                index1 = index0 + 1
                if index1 >= sine_500hz_length:
                    index1 = 0
                    
                frac = phase_accumulators[o] - index0
                
                val0 = sine_500hz_pcm[index0]
                val1 = sine_500hz_pcm[index1]
                
                interpolated = val0 + frac * (val1 - val0)
                mixed_sample += interpolated * osc_amplitudes[o]
                
                phase_accumulators[o] += step_sizes[o]
                while phase_accumulators[o] >= sine_500hz_length:
                    phase_accumulators[o] -= sine_500hz_length
                    
            noise = generate_white_noise() * tire_noise_amp * 32767.0
            mixed_sample += noise
            mixed_sample *= target_amp
            
            # Anti-clipping
            if mixed_sample > 32767: mixed_sample = 32767
            elif mixed_sample < -32768: mixed_sample = -32768
                
            data = struct.pack('<h', int(mixed_sample))
            wav_file.writeframesraw(data)
                
    print(f"Created {output_path} successfully. This 2-second clip simulates EV acceleration from 0 to 20 km/h.")

if __name__ == "__main__":
    main()
