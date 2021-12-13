use std::fmt;
use std::fmt::Error;
use std::fmt::Formatter;
use crate::prng::Xorrng;
use crate::matrix::Matrix;
use std::f64;
use std::fs::File;
use std::io;
use std::io::Write;
use std::io::Read;
#[derive(Debug, PartialEq)]
pub struct Model{
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
}
fn logistic(input: f64) -> f64{
    1.0/(1.0 + f64::exp(-input))
}
fn logistic_deriv(input: f64) -> f64{
    logistic(input) * (1.0 - logistic(input))
}
impl fmt::Display for Model{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error>{
	for i in 0..self.weights.len(){
	    write!(f, "{}", self.biases[i])?;
	    write!(f, "{}", self.weights[i])?;
	}
	write!(f, "{}", self.biases[self.weights.len()])?;
	Ok(())
    }
}
impl Model{
    pub fn new(arch: Vec<usize>) -> Self{
	let mut gen = Xorrng::seed(12345);
	let mut w = Vec::with_capacity(arch.len() - 1);
	let mut b = Vec::with_capacity(arch.len());
	for i in 0..arch.len() - 1{
	    b.push(Matrix::random(arch[i], 1, -1.0, 1.0, &mut gen));
	    w.push(Matrix::random(arch[i+1], arch[i], -1.0, 1.0, &mut gen));
	}
	b.push(Matrix::random(arch[arch.len()-1], 1, -1.0, 1.0, &mut gen));
	Self{
	    weights: w,
	    biases: b,
	}
    }
    pub fn from_file(path: String) -> io::Result<Self>{
	let mut f = File::open(path)?;
	let mut buffer = Vec::new();
	f.read_to_end(&mut buffer)?;
	let mut w = Vec::new();
	let mut b = Vec::new();
	let mut index = 0;
	let mut on_bias = true; 
	while index < buffer.len(){
	    let rows = usize::from_ne_bytes(buffer[index..index+8].try_into().unwrap());
	    let columns = usize::from_ne_bytes(buffer[index+8..index+16].try_into().unwrap());
	    let read_next = rows * columns * 8;
	    let next_matrix = Matrix::from_bytes(buffer[index..index+16+read_next].to_vec());
	    if on_bias {
		b.push(next_matrix);
	    }else{
		w.push(next_matrix);
	    }
	    on_bias = !on_bias;
	    index = index + 16 + read_next;
	}
	Ok(Self{
	    biases: b,
	    weights: w,
	})
    }
    pub fn to_file(&self, path: String) ->  io::Result<()>{
	let mut buffer = File::create(path)?;
	for i in 0..self.weights.len(){
	    buffer.write(&self.biases[i].as_bytes())?;
	    buffer.write(&self.weights[i].as_bytes())?;
	}
	buffer.write(&self.biases[self.biases.len() - 1].as_bytes())?;
	Ok(())
    }
}
impl Model{
    fn forward(&self, input: Matrix) -> Vec<Matrix>{
	//get the_weighted_inputs ie the vector representing the pre-activation signal to a layer
	let mut weighted_inputs = Vec::with_capacity(self.biases.len());
	weighted_inputs.push(input.combine(&self.biases[0], |a, b| a + b));
	for i in 0..self.weights.len(){
	    weighted_inputs.push(self.weights[i].mult(&weighted_inputs[i].apply(logistic)).unwrap().combine(&self.biases[i+1], |a, b| a + b));
	}
	weighted_inputs
    }
    fn output(weighted_inputs: &Vec<Matrix>) -> Matrix{
	//take the final_weighted input and activate it
	weighted_inputs[weighted_inputs.len() - 1].apply(logistic)
    }
    fn cost(output: Matrix, actual: &Matrix) -> f64{
	//take the output, and calculate a cost relative to the actual answer
	0.5 * output.combine(&actual, |a, b| (a - b) * (a - b)).to_scalar(|acc, x| acc + x)
    }
    fn backward(&self, weighted_inputs: &Vec<Matrix>, actual: Matrix) -> Vec<Matrix>{
	//calculate the partial derivative of the cost function with respect to this layer's weighted_input; 
	let mut weighted_errors = weighted_inputs.clone(); //hack to allocate the right space;
	let final_layer_index = weighted_inputs.len() - 1;
	weighted_errors[final_layer_index] = Self::output(weighted_inputs).combine(&actual, |a, b| a - b).combine(&weighted_inputs[final_layer_index].apply(logistic_deriv), |a, b| a * b);
	for i in (0..weighted_errors.len() - 1).rev(){
	    println!("{}", i);
	    weighted_errors[i] = self.weights[i].transpose().mult(&weighted_errors[i+1]).unwrap().combine(&weighted_inputs[i].apply(logistic_deriv), |a, b| a * b);
	}
	weighted_errors.to_vec()
    }
    fn calculate_weight_deltas(weighted_inputs: &Vec<Matrix>, weighted_errors: &Vec<Matrix>) -> Vec<Matrix>{
	//calculate the weight delta based on the weighted_input delta
	let mut weight_deltas = Vec::with_capacity(weighted_inputs.len() - 1);
	for i in 0..weighted_inputs.len() - 1{
	    weight_deltas.push(Matrix::from_outer_product(&weighted_errors[i+1], &weighted_inputs[i].apply(logistic).transpose()));
	}
	weight_deltas
    }
    pub fn do_one_example(&self, input: Matrix, output: Matrix) -> (Vec<Matrix>, Vec<Matrix>){
	//put all the calculation functions in one function
	let weighted_inputs = self.forward(input);
	println!("model output:\n{}", Self::output(&weighted_inputs));
	println!("answer:\n{}", output.transpose());
	println!("cost: {}", Self::cost(Self::output(&weighted_inputs), &output));
	let weighted_errors = self.backward(&weighted_inputs, output);
	let weight_deltas = Self::calculate_weight_deltas(&weighted_inputs, &weighted_errors);
	(weighted_errors, weight_deltas) //goddammit I put weighted_input instead of weighted_errors;
    }
    pub fn update(&mut self, deltas: (Vec<Matrix>, Vec<Matrix>), learning_rate: f64){
	//actually update the state of the model modified by a learning rate	
	for i in 0..self.biases.len(){
	    self.biases[i].combine_mut(&deltas.0[i], |a, b|  a - b * learning_rate);
	}
	for i in 0..self.weights.len(){
	    self.weights[i].combine_mut(&deltas.1[i], |a, b| a - b * learning_rate);
	}
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn file(){
	let k = Model::new(vec![3, 4, 5, 6]);
	k.to_file("test.mdl".to_string()).unwrap();
	let r = Model::from_file("test.mdl".to_string()).unwrap();
	assert_eq!(k, r);
    }
}
