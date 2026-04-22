import nemo.collections.asr as nemo_asr

# Load Canary model
print("Loading model...")
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="nvidia/canary-1b-flash"
)

print("Model loaded!")

# Transcribe audio
audio_file = "test.wav"  # put your file here

print("Transcribing...")
text = asr_model.transcribe([audio_file])

print("\nResult:")
print(text[0])
