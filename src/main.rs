mod layer;
mod neuron;
mod neural_network;
mod dense_layer;
mod activation;
mod network_config;

extern crate csv;
extern crate image;

use std::error::Error;
use std::usize;

use activation::ActivationType;
use csv::Reader;

use crate::activation::Activation;
use crate::neural_network::NeuralNetwork;


use self::network_config::NetworkConfig;

fn main() -> Result<(), Box<dyn Error>>{    

    let mut train_x: Vec<Vec<f64>> = vec![];
    let mut train_y: Vec<Vec<f64>> = vec![];
    let mut test_x: Vec<Vec<f64>> = vec![];
    let mut test_y: Vec<Vec<f64>> = vec![];

    let rdr = Reader::from_path("train.csv")?;
    split_train_and_test(rdr, &mut train_x, &mut train_y, &mut test_x, & mut test_y)?;
    train_network_and_save_to_file(&mut train_x, &mut train_y)?;
    println!("Rede treinada e salva!");


    let mut neural_network = NeuralNetwork::from_file_model("model.bin")?;

    let mut total_count: u16 = 0;
    let mut errors: u16 = 0;

    for (input, prediction) in test_x.iter().zip(test_y.iter()) {
        let nn_predictions: Box<[f64]> = neural_network.predict(input);
        let predicted = max_index(&nn_predictions);
        let target = max_index(&prediction);
        if target != predicted {
            errors += 1;
        }
        total_count +=1;
    }

    println!("Taxa de acertos/Acurácia: {}%", (1f32 - errors as f32 / total_count as f32) * 100f32);

    Ok(())
}

fn train_network_and_save_to_file(train_x: &Vec<Vec<f64>>, train_y: &Vec<Vec<f64>>) -> Result<(), Box<dyn Error>> {
    let config = NetworkConfig {
        activation: Activation { activation_type: ActivationType::Sigmoid },
        output_activation: Activation { activation_type: ActivationType::Sigmoid },
        input_size: 784,
        layers_size: 16,
        layers_quantity: 2,
        output_size: 10,
        learning_rate: 0.05,
    };
    let mut neural_network = NeuralNetwork::new(config);
    neural_network.train_many(train_x, train_y);
    neural_network.save("model.bin")?;
    Ok(())
}

fn split_train_and_test(
    mut rdr: Reader<std::fs::File>,
    train_x: &mut Vec<Vec<f64>>,
    train_y: &mut Vec<Vec<f64>>,
    test_x: &mut Vec<Vec<f64>>,
    test_y: &mut Vec<Vec<f64>>,
) -> Result<(), Box<dyn Error>> {
    
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
            train_x.push(right);
            train_y.push(predictions);
        } else {
            test_x.push(right);
            test_y.push(predictions);
        }
    }
    
    Ok(())
}


fn max_index(vector: &[f64]) -> u8 {
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
