#ifndef AVAS_ALGORITHM_H
#define AVAS_ALGORITHM_H

#include <stdint.h>
#include <stddef.h>

/**
 * Initialize the AVAS algorithm state.
 */
void avas_init(uint32_t sample_rate);

/**
 * Set the current vehicle speed in km/h.
 * The speed is expected to be >= 0.0f.
 * Reverse speeds should be passed as absolute values, as reversing also requires sound.
 */
void avas_set_speed(float speed_kmh);

/**
 * Trigger the vehicle honk/horn.
 * @param is_honking 1 to start honking, 0 to stop.
 */
void avas_set_honk(int is_honking);

/**
 * Generate the next batch of audio samples.
 * @param buffer Output buffer for 16-bit PCM samples.
 * @param num_samples Number of samples to generate.
 */
void avas_generate_samples(int16_t *buffer, size_t num_samples);

#endif // AVAS_ALGORITHM_H
