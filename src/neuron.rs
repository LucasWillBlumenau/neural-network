use crate::layer::Layer;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    pub holded: f64,
    pub bias: f64,
    pub sum: f64,
    pub weights: Vec<f64>,
}

impl Neuron {
    
   
    pub fn new(weights_size: u16) -> Self {
        let mut generator = rand::thread_rng();
        let mut weights: Vec<f64> = vec![];

        let limit = (1.0 / (weights_size as f64).sqrt()).sqrt();
        for _ in 0..weights_size {
            let value: f64 = generator.gen_range(-limit..=limit);
            weights.push(value);
        }

        let holded = 0f64;
        let bias: f64 = generator.gen_range(-limit..=limit);
        let sum = 0f64;

        Neuron { holded, bias, sum, weights }
    }

    
    pub fn compute_sum<T: Layer>(&self, layer: &T) -> f64 {
        let mut sum = self.bias;
        for (holded, weight) in layer.get_holded_values().zip(self.weights.iter()) {
            sum += holded * weight;
        }
        sum
    }
    
}
