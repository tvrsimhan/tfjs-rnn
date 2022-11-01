export function scaled_in(t) {
    return tf.div(tf.add(t, 46), 50);
}

export function scaled_out(t) {
    return tf.div(tf.sub(t, 6), 82);
}

export function inv_scaled_in(t) {
    return tf.sub(tf.mul(t, 50), 46);
}

export function inv_scaled_out(t) {
    return tf.add(tf.mul(t, 82), 6);
}


export function audio_to_magnitude_db_and_phase(data, n_fft, hop_length_fft) {
    // console.log(data.shape);
    data.print();
    console.log(data.shape);
    D = tf.signal.stft(data, n_fft, hop_length_fft);
    D.print();
    mag = tf.abs(D);
    mag.print();
    zeros_to_ones = tf.equal(tf.zerosLike(mag), mag);
    mag_nonzero = tf.add(zeros_to_ones, mag);
    phase = tf.complex(tf.add(tf.div(tf.real(D), mag_nonzero), zeros_to_ones), tf.div(tf.imag(D), mag_nonzero));
    power = tf.square(mag_nonzero);
    ref_value = mag.max().square();
    mag = tf.mul(10.0 * 2.303, power.log());
    mag = tf.sub(mag, tf.mul(10.0 * 2.303, ref_value.log()));

    return [mag, phase, ref_value];
    // To access return value:
    // const [first, second] = getValues();
}

export function numpy_audio_to_matrix_spectrogram(data, dim_sqr_spec, n_fft, hop_length_fft) {
    nb_audio = data.shape[0];
    var m_mag, m_phase;
    console.log(data.shape);
    console.log(data.slice([1], [1]).shape);
    for (var i = 0; i < nb_audio; i++) {
        try {
            [m_mag, m_phase] = audio_to_magnitude_db_and_phase(data.slice([i], [1]), n_fft, hop_length_fft);
            console.log(data.slice([i], [1]));

        } catch (error) {
            console.error(error);
            console.log("Error at audio " + i);
        }
        if (i == 0) {
            mag = m_mag;
            phase = m_phase;
        } else {
            mag = tf.concat([mag, m_mag], 0);
            phase = tf.concat([phase, m_phase], 0);
            console.log('in ' + mag.shape);
        }
    }
    console.log(mag.shape);
    console.log(phase.shape);
    return [mag, phase];
}


export function magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, amp_db, phase, ref_value) {
    amp_db_rev = tf.mul(ref_value.square(), tf.pow(10.0, tf.div(amp_db, 10.0))).sqrt();
    audio_reverse_stft = tf.mul(amp_db_rev, phase);
    audio_reconstruct = tf.signal.inverse_stft(audio_reverse_stft, frame_length, hop_length_fft);
    return audio_reconstruct;
}

export function audio_to_audio_frame_stack(data, frame_length, hop_length_frame) {
    audio_frame_stack = tf.signal.frame(data, frame_length, hop_length_frame);
    audio_frame_stack.print();
    console.log(audio_frame_stack.shape);
    return audio_frame_stack;
}
