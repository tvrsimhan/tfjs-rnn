let dogBarkingBuffer = null
let data
let sample_rate = 8000
let frame_length = 8064
let hop_length_frame = 8064
let n_fft = 255
let hop_length_fft = 63
let dim_sqr_spec = Math.floor(n_fft / 2) + 1

function audio_to_audio_frame_stack(data, frame_length, hop_length_frame) {
	audio_frame_stack = tf.signal.frame(data, frame_length, hop_length_frame)
	audio_frame_stack.print()
	console.log("audio_frame_stack " + audio_frame_stack.shape)
	return audio_frame_stack
}

function inverse_stft(mag, phase, frame_length, hop_length_fft, n_fft) {
	console.log("mag " + mag)
	console.log("phase " + phase)
	phase.print()

	let stft_a = tf.mul(mag, phase)
	// pad_amount = (frame_length - hop_length_fft);
	// data = tf.mirrorPad(data, [[Math.floor(pad_amount / 2) + 1], [0]], 'reflect');
	console.log(data.shape)
	console.log(stft_a.shape)
	// console.log('stft_a ' + stft_a);
	stft_a.print()
	// stft_a.reshape([stft_a.shape[1], stft_a.shape[2], stft_a.shape[0]]);
	real_frames = stft_a.irfft()
	console.log("real_frames " + real_frames.shape)
	real_frames.print()
	real_frames = tf.squeeze(real_frames)
	console.log("real_frames " + real_frames.shape)
	real_frames = real_frames.reshape([
		real_frames.shape[0] * real_frames.shape[1],
	])
	console.log("real_frames " + real_frames.shape)
	// console.log(frame_length);
	real_frames = real_frames.slice([0], frame_length)
	console.log("real_frames " + real_frames.shape)
	// real_frames = real_frames.reshape([1, real_frames.shape[0]]);
	// console.log('real_frames ' + real_frames.shape);
	return real_frames
}

function audio_to_magnitude_db_and_phase(
	data,
	n_fft,
	frame_length,
	hop_length_fft
) {
	// console.log(data.shape);
	// data.print();
	// console.log(data.shape);
	data = data.reshape([data.shape[1], data.shape[0]])
	// console.log(data.shape);
	// console.log(frame_length + ' ' + hop_length_fft);
	pad_amount = frame_length - hop_length_fft
	// console.log(pad_amount);
	data = tf.mirrorPad(
		data,
		[
			[Math.floor(pad_amount / 2) + 1, Math.floor(pad_amount / 2) + 1],
			[0, 0],
		],
		"reflect"
	)
	console.log(data.shape)

	D = tf.signal.stft(data, frame_length, hop_length_fft, n_fft)
	// D.print();
	mag = tf.abs(D)
	// mag.print();
	console.log("mag " + mag.shape)
	zeros_to_ones = tf.equal(tf.zerosLike(mag), mag)
	mag_nonzero = tf.add(zeros_to_ones, mag)
	phase = tf.complex(
		tf.add(tf.div(tf.real(D), mag_nonzero), zeros_to_ones),
		tf.div(tf.imag(D), mag_nonzero)
	)
	power = tf.square(mag_nonzero)
	ref_value = mag.max().square()
	mag = tf.mul(10.0 * 2.303, power.log())
	mag = tf.sub(mag, tf.mul(10.0 * 2.303, ref_value.log()))

	return [mag, phase, ref_value]
	// To access return value:
	// const [first, second] = getValues();
}

function audio_to_matrix_spectrogram(
	data,
	dim_sqr_spec,
	n_fft,
	hop_length_fft
) {
	nb_audio = data.shape[0]
	let m_mag, m_phase
	let mag = [],
		phase = []
	chunks = tf.split(data, nb_audio, 0)
	console.log("chunks " + chunks[0].shape)

	for (let i = 0; i < nb_audio; i++) {
		try {
			;[m_mag, m_phase] = audio_to_magnitude_db_and_phase(
				chunks[i],
				n_fft,
				frame_length,
				hop_length_fft
			)
			console.log(i + " " + m_mag.shape)
		} catch (error) {
			console.error(error)
			console.log("Error at audio " + i)
		}

		mag.push(m_mag)
		phase.push(m_phase)
		console.log("phase shape" + phase[0].shape)
	}
	// mat_mag = tf.stack(mag);
	// = tf.stack(phase);
	// console.log(mat_mag.shape);
	// console.log('phase ' + phase.length);
	return [mag, phase]
}

function magnitude_db_and_phase_to_audio(
	frame_length,
	hop_length_fft,
	amp_db,
	phase,
	ref_value
) {
	amp_db_rev = tf
		.mul(ref_value.square(), tf.pow(10.0, tf.div(amp_db, 10.0)))
		.sqrt()
	// audio_reverse_stft = tf.mul(amp_db_rev, phase);
	phase = tf.stack([phase])
	console.log("phase shape " + phase.shape)

	// console.log('t_phase ' + t_phase.shape);
	audio_reconstruct = inverse_stft(
		amp_db_rev,
		phase,
		frame_length,
		hop_length_fft
	)
	console.log("audio_reconstruct " + audio_reconstruct.shape)
	return audio_reconstruct
}

function matrix_spectrogram_to_audio(
	mag,
	phase,
	frame_length,
	hop_length_fft,
	ref_value
) {
	nb_audio = mag.shape[0]
	let audio_reconstruct_arr = []
	for (let i = 0; i < nb_audio; i++) {
		audio_reconstruct = magnitude_db_and_phase_to_audio(
			frame_length,
			hop_length_fft,
			mag.slice([i], [1]),
			phase[i],
			ref_value
		)
		console.log(audio_reconstruct.shape)
		audio_reconstruct.print()
		console.log("array_reconstruct " + audio_reconstruct.shape)
		audio_reconstruct_arr.push(audio_reconstruct)
	}
	audio_reconstruct_stack = tf.stack(audio_reconstruct_arr)
	console.log("audio_reconstruct_stack " + audio_reconstruct_stack.shape)
	return audio_reconstruct_stack
}

function scaled_in(t) {
	return tf.div(tf.add(t, 46), 50)
}

function scaled_out(t) {
	return tf.div(tf.sub(t, 6), 82)
}

function inv_scaled_in(t) {
	return tf.sub(tf.mul(t, 50), 46)
}

function inv_scaled_out(t) {
	return tf.add(tf.mul(t, 82), 6)
}

async function prediction(
	data,
	dim_sqr_spec,
	frame_length,
	n_fft,
	hop_length_frame,
	hop_length_fft
) {
	const model = await tf.loadLayersModel("/rnn/model.json")
	data = audio_to_audio_frame_stack(data, frame_length, hop_length_frame)
	;[amp, phase] = audio_to_matrix_spectrogram(
		data,
		dim_sqr_spec,
		n_fft,
		hop_length_fft
	)
	amp_db = tf.stack(amp)
	console.log("amp_db shape " + amp_db.shape)
	// console.log('amp_db ' + amp_db);
	// console.log('phase ' + phase);
	console.log("phase shape " + phase.length)
	console.log("phase[0] shape " + phase[0].shape)
	X_in = scaled_in(amp_db)
	console.log("X_in " + X_in.shape)
	X_in = X_in.reshape([X_in.shape[0], X_in.shape[1], X_in.shape[2], 1])
	console.log("X_in " + X_in.shape)
	X_pred = model.predict(X_in)
	console.log("X_pred " + X_pred.shape)
	inv_sca_X_pred = inv_scaled_out(X_pred)
	// console.log('split ' + inv_sca_X_pred.unstack(3)[0].shape);
	X_denoise = amp_db.sub(inv_sca_X_pred.unstack(3)[0])
	console.log("X_denoise " + X_denoise.shape)
	audio_denoise_recons = matrix_spectrogram_to_audio(
		X_denoise,
		phase,
		frame_length,
		hop_length_fft,
		tf.tensor1d([1])
	)
	console.log("audio_denoise_recons " + audio_denoise_recons.shape)
	nb_samples = audio_denoise_recons.shape[0]
	denoise_long = audio_denoise_recons.reshape([1, nb_samples * frame_length])
	console.log("denoise_long " + denoise_long.shape)
	return denoise_long
}

function loadDogSound(url) {
	let request = new XMLHttpRequest()
	request.open("GET", url, true)
	request.responseType = "arraybuffer"

	// Decode asynchronously
	request.onload = function () {
		let context = new AudioContext({ sampleRate: sample_rate })
		let audioData = request.response
		console.log(request.response)
		context.decodeAudioData(request.response, function (buffer) {
			console.log(buffer)
			data = tf.tensor(buffer.getChannelData(0))
			data.print()

			dogBarkingBuffer = buffer
			prediction(
				data,
				dim_sqr_spec,
				frame_length,
				n_fft,
				hop_length_frame,
				hop_length_fft
			)
		})
	}
	request.send()
	console.log(typeof dogBarkingBuffer)
}

function playSound(arrybuffer) {
	context.decodeAudioData(arrybuffer, function (buffer) {
		console.log(buffer)

		dogBarkingBuffer = buffer
		// prediction(data, 209, 209);
	})
	//let source = context.createBufferSource();
	// creates a sound source
	//  context.decodeAudioData(buffer,(buffer)=>{
	//     source.buffer = buffer;
	// },(err)=>console.error("error while decoding"));
	source.buffer = buffer
	console.log(source) // tell the source which sound to play
	source.connect(context.destination) // connect the source to the context's destination (the speakers)
	source.start()
	/*const streamNode = context.createMediaStreamDestination();
      source.connect(streamNode);
      const audioElem = new Audio();
      audioElem.controls = true;
      document.body.appendChild(audioElem);
      audioElem.srcObject = streamNode.stream;                          // play the source now
      source.start();*/
}
