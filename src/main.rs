mod layer;
mod neuron;
mod neural_network;
mod hidden_layer;
mod input_layer;
mod activation;
mod network_config;

extern crate csv;
extern crate image;

use std::error::Error;
use std::usize;

use activation::ActivationType;
use csv::Reader;

use crate::activation::Activation;
use crate::input_layer::InputLayer;
use crate::neural_network::NeuralNetwork;


use self::network_config::NetworkConfig;

fn main() -> Result<(), Box<dyn Error>>{
    
    let activation = Activation { activation_type: ActivationType::Sigmoid };

    let config = NetworkConfig {
        activation,
        input_size: 784,
        layers_size: 10,
        layers_quantity: 1,
        output_size: 10,
        learning_rate: 0.5,
    };
    let mut neural_network = NeuralNetwork::new(config);

    let mut rdr = Reader::from_path("train.csv")?;

    let mut train_data: Vec<(Vec<f64>, Vec<f64>)> = vec![];
    let mut test_data: Vec<(Vec<f64>, Vec<f64>)> = vec![];

    for (i, result) in rdr.records().enumerate() {
        let record = result?;

        let values: Vec<f64> = record.iter()
            .map(|number| number.parse().unwrap())
            .collect();
        let (left, right) = values.split_at(1);
        let index = left[0] as usize;
        
        let mut predictions: Vec<f64> = [0f64; 10].to_vec();
        predictions[index] = 1f64;

        let right: Vec<f64> = right.iter().map(|value| value / 255f64).collect();
        
        if i < 30_000 {
            train_data.push((predictions, right));
        } else {
            test_data.push((predictions, right));
        }
    }

    for (prediction, input) in train_data.iter() {
        neural_network.train(&InputLayer { values: input.clone() }, &prediction);
    }

    neural_network.save("model.bin")?;

    let mut total_count: u16 = 0;
    let mut errors: u16 = 0;

    for (prediction, input) in test_data {
        let nn_predictions = neural_network.predict(&InputLayer { values: input });
        let predicted = max_index(&nn_predictions);
        let target = max_index(&prediction);
        if target != predicted {
            errors += 1;
        }
        total_count +=1;
    }

    println!("Precisão: {}%", (1f32 - errors as f32 / total_count as f32) * 100f32);

    Ok(())
}


fn max_index(vector: &Vec<f64>) -> u8 {
    let mut max_index = 0;
    let mut max_value = vector[0];
    for (i, number) in vector.iter().enumerate() {
        if *number > max_value {
            max_index = i;
            max_value = *number;
        } 
    }

    return max_index as u8;
}
