use std::fs;
use std::io::{Read, Write};

use bincode::Error;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::dense_layer::DenseLayer;
use crate::layer::Layer;
use crate::network_config::NetworkConfig;

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork {
    activation: Activation,
    output_activation: Activation,
    hidden_layers: Vec<DenseLayer>,
    output_layer: DenseLayer,
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
        
        hidden_layers.push(DenseLayer::new(config.layers_size, config.input_size));
        
        for _ in 1..config.layers_quantity {
            hidden_layers.push(DenseLayer::new(config.layers_size, config.layers_size));
        }
       
        let activation = config.activation;
        let output_activation = config.output_activation;
        let output_layer = DenseLayer::new(config.output_size, config.layers_size);
        let learning_rate = config.learning_rate;
        let input_size = config.input_size;
        let output_size = config.output_size;

        NeuralNetwork {
            activation,
            output_activation,
            hidden_layers,
            output_layer,
            learning_rate,
            input_size,
            output_size,
        }
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

    pub fn predict(&mut self, input: &[f64]) -> Box<[f64]> {
        assert_eq!(input.len() as u16, self.input_size);
        
        self.feed_foward(&input);

        let predictions: Box<[f64]> = self.output_layer.neurons.iter()
                                                               .map(|neuron| neuron.holded)
                                                               .collect();
        predictions
    }

    pub fn train_many(&mut self, inputs: &Vec<Vec<f64>>, predictions: &Vec<Vec<f64>>) {
        assert_eq!(inputs.len(), predictions.len());

        for (input, prediction) in inputs.iter().zip(predictions.iter()) {
            self.train(input, prediction);
        }
    }

    pub fn train(&mut self, input: &[f64], predictions: &[f64]) {
        assert_eq!(input.len() as u16, self.input_size);
        assert_eq!(predictions.len() as u16, self.output_size);

        self.feed_foward(input);
        self.backpropagate(input, predictions);
    }

    fn backpropagate(&mut self, input: &[f64], predictions: &[f64]) {
        for (neuron, prediction) in self.output_layer.neurons.iter_mut().zip(predictions.iter()) {
            let error_derivative = 2f64 * (neuron.holded - prediction) * self.output_activation.derivative(neuron.sum);

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

                let prev_layer_values = prev_layer.neurons.iter()
                                                                                     .map(|neuron| neuron.holded);
                for (weight, activation) in neuron.weights.iter_mut().zip(prev_layer_values) {
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

        let input = Layer::InputLayer(input);
        for (i, neuron) in self.hidden_layers[0].neurons.iter_mut().enumerate() {
            let error_derivative = errors[i] * self.activation.derivative(neuron.sum);
            let delta = error_derivative * self.learning_rate;
            neuron.bias -= delta;

            
            for (weight, activation) in neuron.weights.iter_mut().zip(input.get_values()) {
                *weight -= activation * delta;
            }
        }
    }

    fn feed_foward(&mut self, input: &[f64]) {
        
        let (first, rest) = self.hidden_layers.split_at_mut(1);
        
        let first_layer = &mut first[0];
        
        for neuron in &mut first_layer.neurons {
            neuron.sum = neuron.compute_sum(Layer::InputLayer(input));
            neuron.holded = self.activation.function(neuron.sum);
        }
        
        let mut last_layer = first_layer;
        
        for layer in rest {
            for neuron in &mut layer.neurons {
                neuron.sum = neuron.compute_sum(Layer::DenseLayer(last_layer));
                neuron.holded = self.activation.function(neuron.sum);
            }
            last_layer = layer;
        }
        
        for neuron in self.output_layer.neurons.iter_mut() {
            neuron.sum = neuron.compute_sum(Layer::DenseLayer(last_layer));
            neuron.holded = self.output_activation.function(neuron.sum);
        }
    }
    
}
