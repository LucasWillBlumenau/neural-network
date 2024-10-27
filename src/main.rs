mod layer;
mod neuron;
mod neural_network;
mod hidden_layer;
mod input_layer;
mod activation;


extern crate csv;

use std::error::Error;
use std::f64::consts::E;
use std::usize;

use csv::Reader;

use crate::activation::Activation;
use crate::input_layer::InputLayer;
use crate::neural_network::NeuralNetwork;

fn main() -> Result<(), Box<dyn Error>>{
    
    let activation = Activation { function: sigmoid, derivative: sigmoid_derivative };
    let mut neural_network = NeuralNetwork::new(activation, 784, 2, 2, 10);

    let mut rdr = Reader::from_path("train.csv")?;
    let mut count = 0;
    for result in rdr.records() {
        let record = result?;

        let values: Vec<f64> = record.iter()
                                     .map(|number| number.parse().unwrap())
                                     .collect();
        let (left, right) = values.split_at(1);
        let index = left[0] as usize;

        let mut predictions: Vec<f64> = [0f64; 10].to_vec();
        predictions[index] = 1f64;

        let train = InputLayer { values: right.to_vec() };

        if count == 500 {
            let nn_predictions = neural_network.predict(&train);
            println!("{predictions:?}");
            println!("{nn_predictions:?}");
            count = 0;
        }

        neural_network.train(&train, &predictions);

        if count == 0 {
            let loss = neural_network.get_average_loss();
            println!("Average loss: {loss}");
            neural_network.reset_loss();
        }

        count += 1;

    } 

    let i1 = InputLayer { values: vec![12f64, 12f64] };
    let prediction = vec![0f64, 1f64];

    neural_network.train(&i1, &prediction);
    Ok(())
}

fn relu(input: f64) -> f64 {
    if input > 0.0 {
        input
    } else {
        0.0
    }
}

fn relu_derivative(input: f64) -> f64 {
    if input > 0.0 {
        1.0
    } else {
        0.0
    }  
}

