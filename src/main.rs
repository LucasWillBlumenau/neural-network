mod layer;
mod neuron;
mod neural_network;
mod hidden_layer;
mod input_layer;
mod activation;
mod network_config;

extern crate csv;

use std::error::Error;
use std::usize;

use csv::Reader;

use crate::activation::Activation;
use crate::input_layer::InputLayer;
use crate::neural_network::NeuralNetwork;

use self::network_config::NetworkConfig;

fn main() -> Result<(), Box<dyn Error>>{
    
    let activation = Activation::sigmoid();

    let config = NetworkConfig {
        activation,
        input_size: 784,
        layers_size: 16,
        layers_quantity: 2,
        output_size: 10,
        learning_rate: 0.1,
    };
    let mut neural_network = NeuralNetwork::new(config);

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

        let right: Vec<f64> = right.iter().map(|value| value / 255f64).collect();
        let train = InputLayer { values: right };
        if train.values.len() != 784 {
            println!("{:?}", train.values);
        }


        neural_network.train(&train, &predictions);

        if count == 5000 {
            let loss = neural_network.get_average_loss();
            let nn_predictions = neural_network.predict(&train);
            println!("Average loss: {loss}");
            println!("Predicitions: {predictions:?}");
            println!("Network Predictions: {nn_predictions:?}");
            neural_network.reset_loss();
            count = 0;
        }

        count += 1;

    } 
    Ok(())
}

