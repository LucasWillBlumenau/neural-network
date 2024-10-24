use crate::activation::Activation;
use crate::hidden_layer::HiddenLayer;
use crate::input_layer::InputLayer;


#[derive(Debug)]
pub struct NeuralNetwork {
    activation: Activation,
    hidden_layers: Vec<HiddenLayer>,
    output_layer: HiddenLayer,
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

        NeuralNetwork { activation, hidden_layers, output_layer }
    }

    pub fn train(&mut self) {
        
    }

    pub fn backpropagate(&mut self, predictions: &Vec<f32>) {
        let mut costs: Vec<f32> = vec![];
        for (neuron, prediction) in self.output_layer.neurons.iter().zip(predictions.iter()) {
            let mut cost = prediction - neuron.holded;
            cost *= cost;
            costs.push(cost);
        }
        
        
    }

    pub fn feed_foward(&mut self, input: &InputLayer) {
        
        let (first, rest) = self.hidden_layers.split_at_mut(1);
        
        let first_layer = &mut first[0];
        
        for neuron in &mut first_layer.neurons {
            neuron.holded = neuron.activate(input, &self.activation);
        }
        
        let mut last_layer = first_layer;
        
        for layer in rest {
            for neuron in &mut layer.neurons {
                neuron.holded = neuron.activate(last_layer, &self.activation);
            }
            last_layer = layer;
        }
        
        for neuron in &mut self.output_layer.neurons {
            neuron.holded = neuron.activate(last_layer, &self.activation);
        }
    }
    
}
