mod layer;
mod neuron;
mod neural_network;
mod hidden_layer;
mod input_layer;
mod activation;

use crate::activation::Activation;
use crate::input_layer::InputLayer;
use crate::neural_network::NeuralNetwork;

fn main() {
    
    let activation = Activation { function: relu, derivative: relu_derivative };
    let mut neural_network = NeuralNetwork::new(activation, 16, 2, 8, 10);
    
    
    let i1 = InputLayer { values: vec![12f32, 12f32, 12f32, 12f32, 234f32, 345f32, 234f32, 12f32] };
    let i2 = InputLayer { values: vec![13f32, 234f32, 1232f32, 1232f32, 1234f32, 345f32, 234f32, 12f32] };
    let i3 = InputLayer { values: vec![32f32, 12f32, 12f32, 12f32, 234f32, 345f32, 234f32, 12f32] };
    
    neural_network.feed_foward(&i1);
    println!("{neural_network:?}");
    neural_network.feed_foward(&i2);
    println!("{neural_network:?}");
    neural_network.feed_foward(&i1);
    println!("{neural_network:?}");
    neural_network.feed_foward(&i2);
    println!("{neural_network:?}");
    neural_network.feed_foward(&i1);
    println!("{neural_network:?}");
    neural_network.feed_foward(&i3);
    println!("{neural_network:?}");

}

fn relu(input: f32) -> f32 {
    if input > 0.0 {
        input
    } else {
        0.0
    }
}

fn relu_derivative(input: f32) -> f32 {
    if input > 0.0 {
        1.0
    } else {
        0.0
    }  
}







