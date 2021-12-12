use std::fmt;
use std::fmt::Error;
use std::fmt::Formatter;
use crate::prng::Xorrng;
use crate::matrix::Matrix;
use std::f64;
#[derive(Debug)]
pub struct Model{
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
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
	    b.push(Matrix::random(arch[i], 1, 0.0, 1.0, &mut gen));
	    w.push(Matrix::random(arch[i+1], arch[i], 0.0, 1.0, &mut gen));
	}
	b.push(Matrix::random(arch[arch.len()-1], 1, 0.0, 1.0, &mut gen));
	Self{
	    weights: w,
	    biases: b,
	}
    }
}
impl Model{
    fn forward(&self, input: Matrix) -> Vec<Matrix>{
	let mut weighted_inputs = Vec::with_capacity(self.biases.len());
	weighted_inputs.push(input.combine(&self.biases[0], |a, b| a + b));
	for i in 0..self.weights.len(){
	    weighted_inputs[i + 1] = self.weights[i].mult(&weighted_inputs[i].apply(logistic)).combine(&self.biases[i+1], |a, b| a + b);
	}
	weighted_inputs
    }
    fn output(weighted_inputs: &Vec<Matrix>) -> Matrix{
	weighted_inputs[weighted_inputs.len() - 1].apply(logistic)
    }
    fn cost(output: Matrix, actual: &Matrix) -> f64{
	0.5 * output.combine(&actual, |a, b| a - b).to_scalar(|acc, x| acc + x)
    }
    fn backward(&self, weighted_inputs: &Vec<Matrix>, actual: Matrix) -> Vec<Matrix>{
	let mut weighted_errors = Vec::with_capacity(weighted_inputs.len());
	let final_layer_index = weighted_errors.len() - 1;
	weighted_errors[final_layer_index] = Self::output(weighted_inputs).combine(&actual, |a, b| a - b).combine(&weighted_inputs[final_layer_index].apply(logistic_deriv), |a, b| a * b);
	for i in (0..weighted_errors.len() - 1).rev(){
	    weighted_errors[i] = weighted_errors[i + 1].mult(&self.weights[i].transpose()).combine(&weighted_inputs[i].apply(logistic_deriv), |a, b| a * b);
	}
	weighted_errors
    }
    fn calculate_weight_deltas(weighted_inputs: &Vec<Matrix>, weighted_errors: &Vec<Matrix>) -> Vec<Matrix>{
	let mut weight_deltas = Vec::with_capacity(weighted_inputs.len() - 1);
	for i in 0..weight_deltas.len(){
	    weight_deltas[i] = Matrix::from_outer_product(&weighted_errors[i+1], &weighted_inputs[i].apply(logistic).transpose());
	}
	weight_deltas
    }
    pub fn do_one_example(&self, input: Matrix, output: Matrix) -> (Vec<Matrix>, Vec<Matrix>){
	let weighted_inputs = self.forward(input);
	println!("{}", Self::output(&weighted_inputs));
	println!("cost: {}", Self::cost(Self::output(&weighted_inputs), &output));
	let weighted_errors = self.backward(&weighted_inputs, output);
	let weight_deltas = Self::calculate_weight_deltas(&weighted_inputs, &weighted_errors);
	(weighted_inputs, weight_deltas)
    }
    pub fn update(&mut self, deltas: (Vec<Matrix>, Vec<Matrix>)){
	for i in 0..self.biases.len(){
	    self.biases[i].combine_mut(&deltas.0[i], |a, b| a - b);
	}
	for i in 0..self.weights.len(){
	    self.weights[i].combine_mut(&deltas.1[i], |a, b| a - b);
	}
    }
}
