use bincode::Error;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::hidden_layer::HiddenLayer;
use crate::input_layer::InputLayer;
use crate::layer::Layer;
use crate::network_config::NetworkConfig;
use std::f64::consts::E;
use std::fs;
use std::io::{Read, Write};


#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork {
    activation: Activation,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: HiddenLayer,
    learning_rate: f64,
    input_size: u16,
    output_size: u16,
}

impl NeuralNetwork {

    pub fn new(config: NetworkConfig) -> Self {
        
        assert_ne!(config.layers_size, 0);
        assert_ne!(config.layers_quantity, 0);
        assert_ne!(config.input_size, 0);
        assert_ne!(config.output_size, 0);

        let mut hidden_layers = vec![];
        
        hidden_layers.push(HiddenLayer::new(config.layers_size, config.input_size));
        
        for _ in 1..config.layers_quantity {
            hidden_layers.push(HiddenLayer::new(config.layers_size, config.layers_size));
        }
       
        let activation = config.activation;
        let output_layer = HiddenLayer::new(config.output_size, config.layers_size);
        let learning_rate = config.learning_rate;
        let input_size = config.input_size;
        let output_size = config.output_size;

        NeuralNetwork { activation, hidden_layers, output_layer, learning_rate, input_size, output_size }
    }

    pub fn save(&self, path: &str) -> Result<(), Error> {

        let encoded: Vec<u8> = bincode::serialize(self)?;
        let mut file = fs::OpenOptions::new().write(true)
                                                   .create(true) 
                                                   .open(path)?;
        file.write_all(&encoded)?;

        Ok(())
    }

    pub fn from_file_model(path: &str) -> Result<Self, Error> {
        let mut file = fs::File::open(path)?;
        let mut buffer: Vec<u8> = vec![];
        file.read_to_end(&mut buffer)?;
        
        let neural_network: Self = bincode::deserialize(&buffer)?;
        Ok(neural_network)
    }

    pub fn predict(&mut self, input: &InputLayer) -> Vec<f64> {
        assert_eq!(input.values.len() as u16, self.input_size);
        
        self.feed_foward(input);

        let predictions: Vec<f64> = self.output_layer.get_holded_values()
                                                     .into_iter()
                                                     .collect();
        predictions
    }

    pub fn train(&mut self, input: &InputLayer, predictions: &Vec<f64>) {
        assert_eq!(input.values.len() as u16, self.input_size);
        assert_eq!(predictions.len() as u16, self.output_size);

        self.feed_foward(input);
        self.backpropagate(input, predictions);
    }

    fn backpropagate(&mut self, input: &InputLayer, predictions: &Vec<f64>) {
        for (neuron, prediction) in self.output_layer.neurons.iter_mut().zip(predictions.iter()) {
            let error_derivative = 2f64 * (neuron.holded - prediction) * neuron.holded * (1f64 - neuron.holded);

            neuron.bias -= error_derivative * self.learning_rate;

            let weigths_iterator = neuron.weights.iter_mut();
            let prev_later_iterator = self.hidden_layers.last().unwrap().neurons.iter();

            for (weight, neuron) in weigths_iterator.zip(prev_later_iterator) {
                *weight -= neuron.holded * error_derivative * self.learning_rate;
            }

            neuron.holded = error_derivative;
        }

        let mut errors: Vec<f64> = vec![];
        errors.reserve_exact(self.output_layer.neurons.len());

        for i in 0..self.output_layer.neurons[0].weights.len() {
            let mut error = 0f64;
            for neuron in self.output_layer.neurons.iter() {
                error += neuron.weights[i] * neuron.holded;
            }
            errors.push(error);
        }

        for i in (1..self.hidden_layers.len()).rev() {
            let (left, right) = self.hidden_layers.split_at_mut(i);

            let current_layer = &mut right[0];
            let prev_layer = left.last().unwrap();

            for (j, neuron) in current_layer.neurons.iter_mut().enumerate() {
                let error_derivative = errors[j] * self.activation.derivative(neuron.sum);
                let delta = error_derivative * self.learning_rate;
                neuron.bias -= delta;

                for (weight, activation) in neuron.weights.iter_mut().zip(prev_layer.get_holded_values()) {
                    *weight -= activation * delta;
                }

                neuron.holded = error_derivative;
            }

            for j in 0..current_layer.neurons[0].weights.len() {
                let mut error = 0f64;
                for neuron in current_layer.neurons.iter() {
                    error += neuron.holded * neuron.weights[j];
                }
                errors[j] = error;
            }
        }

        for (i, neuron) in self.hidden_layers[0].neurons.iter_mut().enumerate() {
            let error_derivative = errors[i] * self.activation.derivative(neuron.sum);
            let delta = error_derivative * self.learning_rate;
            neuron.bias -= delta;

            for (weight, activation) in neuron.weights.iter_mut().zip(input.get_holded_values()) {
                *weight -= activation * delta;
            }
        }
    }

    fn feed_foward(&mut self, input: &InputLayer) {
        
        let (first, rest) = self.hidden_layers.split_at_mut(1);
        
        let first_layer = &mut first[0];
        
        for neuron in &mut first_layer.neurons {
            neuron.sum = neuron.compute_sum(input);
            neuron.holded = self.activation.function(neuron.sum);
        }
        
        let mut last_layer = first_layer;
        
        for layer in rest {
            for neuron in &mut layer.neurons {
                neuron.sum = neuron.compute_sum(last_layer);
                neuron.holded = self.activation.function(neuron.sum);
            }
            last_layer = layer;
        }
        
        for neuron in self.output_layer.neurons.iter_mut() {
            neuron.sum = neuron.compute_sum(last_layer);
            neuron.holded = 1f64 / (1f64 + E.powf(-neuron.sum));
        }
    }
    
}

