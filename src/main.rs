use nn::model::Model;
use nn::model;
use nn::dataset::Image;
use nn::dataset;
use std::io;
fn main(){
    let mut model = Model::new(vec![784, 10]);
    let mut starting = 0;
    let d = dataset::read().unwrap();
    loop{
	let command = user_input().unwrap();
	match &*command{
	    "1" => gen_model(&mut model),
	    "2" => load_model(&mut model).unwrap(),
	    "3" => check_model(&model, &d),
	    "4" => train(&mut starting, &d, &mut model).unwrap(),
	    "5" => {
		save_model(&model).unwrap(); 
		break;
	    },
	    _ => (),
	};
    }
}
fn user_input() -> io::Result<String> {
    println!("Press [1] to generate new model.");
    println!("Press [2] to load old model.");
    println!("Press [3] to check model results.");
    println!("Press [4] to train model.");
    println!("Press [5] to quit.");
    let mut command = String::new();
    io::stdin().read_line(&mut command)?;
    Ok(command.trim().into())
}
fn train(starting: &mut usize, d: &Vec<(u8, Image)>, model: &mut Model) -> io::Result<()>{
    println!("train for how many iterations?");
    let mut iterations = String::new();
    io::stdin().read_line(&mut iterations)?;
    let batch_size = 100;
    let iterations: usize = iterations.trim().parse().unwrap();
    let mut model_copy =  model.clone();
    for i in 0..iterations{
	let index = (i + *starting) % 60000;
	//let index = *starting;
	//println!("{}", index);
	let pair = dataset::translate(&d[index]);
	if i % batch_size != batch_size - 1{
	    let deltas = model.do_one_example(pair.0, pair.1);
	    model_copy.update(deltas, 0.01 / (batch_size as f64));
	}else{ 
	    *model = model_copy;
	    model_copy = model.clone();
	}
    }
    *starting += iterations;
    Ok(())
}
fn gen_model(model: &mut Model){
    *model = Model::new(vec![784, 10]);
}
fn load_model(model: &mut Model) -> io::Result<()>{
    println!("Please input path to the model");
    let mut path = String::new();
    io::stdin().read_line(&mut path)?;
    path = path.trim().to_string();
    *model = Model::from_file(path)?;
    Ok(())
}
fn save_model(model: &Model) -> io::Result<()>{
     println!("Please input path to the model");
    let mut path = String::new();
    io::stdin().read_line(&mut path)?;
    path = path.trim().to_string();
    model.to_file(path)?;
    Ok(())
}
fn check_model(model: &Model, d: &Vec<(u8, Image)>){
    for i in 0..60000{
	println!("{}\nActual answer:{}", d[i].1, d[i].0);
	let pair = dataset::translate(&d[i]);
	let output = model.forward(pair.0);
	let confidence = output[output.len() - 1].apply(model::logistic);
	let mut max = 0.0;
	let mut ans = 0;
	for i in 0..10{
	    if confidence.data[i][0] > max{
		max = confidence.data[i][0];
		ans = i;
	    }
	}
	println!("{}: confidence {1:.5}%", ans, max * 100.0); 
	loop{
	    println!("press [n] for next");
	    let mut command = String::new();
	    io::stdin().read_line(&mut command).unwrap();
	    command = command.trim().to_string();
	    if command == "n"{
		break;
	    }
	}
    }
}
