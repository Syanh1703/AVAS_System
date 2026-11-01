#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "avas_algorithm.h"

// WAV Header structure (pragma pack ensures exact byte layout)
#pragma pack(push, 1)
typedef struct {
    char riff[4];
    uint32_t overall_size;
    char wave[4];
    char fmt_chunk_marker[4];
    uint32_t length_of_fmt;
    uint16_t format_type;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byterate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_chunk_header[4];
    uint32_t data_size;
} WavHeader;
#pragma pack(pop)

void write_wav_header(FILE *f, uint32_t num_samples, uint32_t sample_rate) {
    WavHeader header = {
        {'R', 'I', 'F', 'F'},
        36 + num_samples * 2,
        {'W', 'A', 'V', 'E'},
        {'f', 'm', 't', ' '},
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        {'d', 'a', 't', 'a'},
        num_samples * 2
    };
    fwrite(&header, 1, sizeof(header), f);
}

int main(void) {
    const char *output_file = "avas_output_c_custom.wav";
    FILE *f = fopen(output_file, "wb");
    if (!f) {
        printf("Failed to open %s for writing.\n", output_file);
        return 1;
    }

    uint32_t sample_rate = 33100;
    uint32_t duration_seconds = 2; // Simulate a 2-second acceleration clip
    uint32_t total_samples = duration_seconds * sample_rate;
    
    // Write WAV header using the known total size
    write_wav_header(f, total_samples, sample_rate);

    // Initialize our AVAS Algorithm state
    avas_init(sample_rate);

    // Process audio in chunks of 1600 samples (0.1 seconds)
    size_t chunk_size = 1600;
    int16_t *buffer = (int16_t *)malloc(chunk_size * sizeof(int16_t));

    if (!buffer) {
        printf("Memory allocation failed.\n");
        fclose(f);
        return 1;
    }

    printf("Simulating %d seconds of EV acceleration...\n", duration_seconds);

    // Loop through the simulation time
    for (size_t i = 0; i < total_samples; i += chunk_size) {
        // Generate a STABLE input sound at 5 km/h (base pitch). 
        // This ensures the volume and pitch are perfectly flat (no highs/lows)
        // so that the HTML UI can do all the dynamic pitching and volume scaling.
        float current_speed = 5.0f;
        
        // 1. Pass the stable vehicle speed into the algorithm
        avas_set_speed(current_speed);
        
        // 2. Generate the next batch of audio samples
        avas_generate_samples(buffer, chunk_size);
        
        // 3. Write out the mixed EV sound samples to the WAV file
        fwrite(buffer, sizeof(int16_t), chunk_size, f);
    }

    free(buffer);
    fclose(f);

    printf("Successfully wrote generated EV sound to %s\n", output_file);
    return 0;
}
