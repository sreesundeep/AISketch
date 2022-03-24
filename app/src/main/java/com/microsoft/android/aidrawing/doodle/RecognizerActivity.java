package com.microsoft.android.aidrawing.doodle;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.microsoft.android.aidrawing.R;


import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class RecognizerActivity extends Activity {
  static final String TAG = "DOODLE";
  static final String MODEL_FILENAME = "model.tflite";
  static final int THREAD_NUM = 4;
  static final int NUMBER_CLASSES = 10;
  static final int IMAGE_PIXELS = 784;
  static final int IMAGE_HEIGHT = 28;
  static final int IMAGE_WIDTH = 28;
  static final int IMAGE_CHANNEL = 1;
  static final int BATCH = 1;
  static final String[] LABELS = {
    "apple", "bed", "cat", "dog", "eye",
    "fish", "grass", "hand", "ice creame", "jacket",
  };

  protected Interpreter interpreter;
  protected Button recognize;
  protected Button clear;
  protected TextView result;
  protected CanvasView canvas;

  private float[][][][] image = null;
  private float[][] probabilities = null;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_recognizer);

    canvas = (CanvasView) findViewById(R.id.canvas);
    recognize = (Button) findViewById(R.id.recognize);
    result = (TextView) findViewById(R.id.result);
    recognize.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        // Fill input pixels.
        int pixels[] = canvas.getPixels();
        assert(pixels.length == IMAGE_PIXELS);
        for(int b = 0; b < BATCH; ++b){
          for(int h = 0; h < IMAGE_HEIGHT; ++h){
            for(int w = 0; w < IMAGE_WIDTH; ++w){
              for(int c = 0; c < IMAGE_CHANNEL; ++c){
                image[b][h][w][c] = Color.alpha(pixels[
                  c
                  + (IMAGE_CHANNEL * w)
                  + (IMAGE_CHANNEL * IMAGE_WIDTH * h)
                  + (IMAGE_CHANNEL * IMAGE_WIDTH * IMAGE_HEIGHT * b)
                ]) / 255.f;
              }
            }
          }
        }

        // Run inference.
        long startTime = System.nanoTime();
        interpreter.run(image, probabilities);
        long endTime = System.nanoTime();
        double ms = (endTime - startTime) / 1000000.0;

        // Show prediction result.
        result.setText("");
        print(String.format("Inference Time: %10f(ms)", ms));
        float[] prob = probabilities[0];
        int classes = argMax(prob);
        print(String.format("Predict: %s (%3.2f%%)",
              LABELS[classes], 100.f * prob[classes]));
        print("Details:");
        for(int i = 0; i < prob.length; ++i) {
          print(String.format("- %11s: %7.4f%%", LABELS[i], 100.f * prob[i]));
        }
      }
    });
    clear = (Button) findViewById(R.id.clear);
    clear.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        canvas.clear();
      }
    });

    // for Input/Output tensor.
    image = new float[BATCH][IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNEL];
    probabilities = new float[BATCH][NUMBER_CLASSES];

    // Initialize the model
    try {
      AssetManager assets = getAssets();
      MappedByteBuffer model = loadAssetToMemory(assets, MODEL_FILENAME);
      Interpreter.Options options = new Interpreter.Options();
      options.setUseNNAPI(true);
      interpreter = new Interpreter(model, options);
      recognize.setEnabled(true);
      print("The model was loaded successful.");
    } catch(IOException e) {
      print("Failed to load the model: " + e.getMessage() + "\n");
      return;
    }
  }

  private void print(String text){
    result.setText(result.getText() + text + "\n");
  }

  private int argMax(float[] array) {
    int index = 0;
    float max = array[0];
    for(int i = 1; i < array.length; ++i) {
      if(max < array[i]){
        max = array[i];
        index = i;
      }
    }
    return index;
  }

  private MappedByteBuffer loadAssetToMemory(AssetManager assets, String path) throws IOException {
    AssetFileDescriptor file = assets.openFd(path);
    FileInputStream stream = new FileInputStream(file.getFileDescriptor());
    FileChannel channel = stream.getChannel();
    long startOffset = file.getStartOffset();
    long declaredLength = file.getDeclaredLength();
    return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
