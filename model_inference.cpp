#include "model_weights.h"

void forward_pass(float input[6], float* output_scores) {
  // normalize input
  float normalized[6];
  for (int i = 0; i < 6; i++) {
    normalized[i] = (input[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
  }

  // layer 1: 6 -> 16, ReLU
  float layer1[16];
  for (int i = 0; i < 16; i++) {
    float sum = BIAS1[i];
    for (int j = 0; j < 6; j++) {
      sum += normalized[j] * W1[i][j];
    }
    layer1[i] = (sum > 0) ? sum : 0;
  }

  // layer 2: 16 -> 8, ReLU
  float layer2[8];
  for (int i = 0; i < 8; i++) {
    float sum = BIAS2[i];
    for (int j = 0; j < 16; j++) {
      sum += layer1[j] * W2[i][j];
    }
    layer2[i] = (sum > 0) ? sum : 0;
  }

  // layer 3: 8 -> 3 (output scores, no activation)
  for (int i = 0; i < 3; i++) {
    float sum = BIAS3[i];
    for (int j = 0; j < 8; j++) {
      sum += layer2[j] * W3[i][j];
    }
    output_scores[i] = sum;
  }
}

int predict_class(float input[6]) {
  float scores[3];
  forward_pass(input, scores);

  int best_idx = 0;
  float best_score = scores[0];
  for (int i = 1; i < 3; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }
  return best_idx;
}