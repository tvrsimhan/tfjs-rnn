context = new AudioContext()
file = new Audio()
file.src = "noisy_voice_long_t1.mp3"
let source = context.createMediaElementSource(file)
source.connect(context.destination)
play.onclick = function () {
  context.resume()
  file.play()
}