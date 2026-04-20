import tensorflow as tf

# 1. Load your trained .keras model
model = tf.keras.models.load_model('unet_brain_tumor.keras', compile=False)

# 2. Initialize the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply Float16 Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# 4. Convert the model
tflite_fp16_model = converter.convert()

# 5. Save to disk
with open('unet_brain_tumor_fp16.tflite', 'wb') as f:
    f.write(tflite_fp16_model)

print("Conversion complete. Float16 TFLite model saved.")