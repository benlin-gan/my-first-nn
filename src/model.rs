use std::fmt;
use std::fmt::Error;
use std::fmt::Formatter;
use crate::prng::Xorrng;
use crate::matrix::Matrix;
#[derive(Debug)]
pub struct Model{
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
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
