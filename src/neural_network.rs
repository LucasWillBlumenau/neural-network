use crate::activation::Activation;
use crate::hidden_layer::HiddenLayer;
use crate::input_layer::InputLayer;
use crate::layer::Layer;
use std::f64::consts::E;


#[derive(Debug)]
pub struct NeuralNetwork {
    activation: Activation,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: HiddenLayer,
    learning_rate: f64,
    loss: f64,
    count: u64,
}

impl NeuralNetwork {

    pub fn new(
        activation: Activation,
        layers_size: u16,
        layers_quantity: u8,
        input_size: u16,
        output_size: u16,
    ) -> Self {
        
        let mut hidden_layers = vec![];
        
        hidden_layers.push(HiddenLayer::new(layers_size, input_size));
        
        for _ in 1..layers_quantity {
            hidden_layers.push(HiddenLayer::new(layers_size, layers_size));
        }
        
        let output_layer = HiddenLayer::new(output_size, layers_size);
        let learning_rate = 0.5;
        let loss = 0f64;
        let count = 0;

        NeuralNetwork { activation, hidden_layers, output_layer, learning_rate, loss, count }
    }

    pub fn predict(&mut self, input: &InputLayer) -> Vec<f64> {
        self.feed_foward(input);

        let predictions: Vec<f64> = self.output_layer.get_holded_values()
                                                     .into_iter()
                                                     .collect();
        predictions
    }

    pub fn get_average_loss(&self) -> f64 {
        self.loss / self.count as f64
    }

    pub fn reset_loss(&mut self) {
        self.loss = 0f64;
        self.count = 0;
    }

    pub fn train(&mut self, input: &InputLayer, predictions: &Vec<f64>) {
        self.feed_foward(input);
        self.backpropagate(input, predictions);
    }

    fn backpropagate(&mut self, input: &InputLayer, predictions: &Vec<f64>) {
        for (neuron, prediction) in self.output_layer.neurons.iter_mut().zip(predictions.iter()) {
            self.loss += (neuron.holded - prediction).powi(2);
            self.count += 1;

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
                let error_derivative = errors[j] * (self.activation.derivative)(neuron.sum);
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
            let error_derivative = errors[i] * (self.activation.derivative)(neuron.sum);
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
            neuron.activate(input, self.activation.function);
        }
        
        let mut last_layer = first_layer;
        
        for layer in rest {
            for neuron in &mut layer.neurons {
                neuron.activate(last_layer, self.activation.function);
            }
            last_layer = layer;
        }
        
        let sigmoid = |number: f64| 1f64 / (1f64 + E.powf(-number));
        for neuron in &mut self.output_layer.neurons {
            neuron.activate(last_layer, sigmoid);
        }
    }
    
}

