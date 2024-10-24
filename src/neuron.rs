use crate::layer::Layer;
use crate::activation::Activation;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    pub holded: f32,
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    
    pub fn new(weights_size: u16) -> Self {
        let mut generator = rand::thread_rng();
        let mut weights: Vec<f32> = vec![];

        for _ in 0..weights_size {
            let value: f32 = generator.gen_range(-1.0..=1.0);
            weights.push(value);
        }
        
        let holded = 0f32;
        let bias: f32 = generator.gen_range(-1.0..=1.0);
        
        Neuron { holded, bias, weights }
        
    }
    
    pub fn activate<T: Layer>(&mut self, layer: &T, activation: &Activation) -> f32 {
        let mut sum = 0f32;
        for (holded, weight) in layer.get_holded_values().zip(self.weights.iter()) {
            sum += holded * weight;
        }
        (activation.function)(sum)
    }
    
}
