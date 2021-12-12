use crate::prng::Xorrng;
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Error;
#[derive(Debug, PartialEq)]
pub struct Matrix{
    data: Vec<Vec<f64>>,
}
impl fmt::Display for Matrix{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error>{
	for i in &self.data{
	    for j in i{
		write!(f, "{} ", j)?;
	    }
	    writeln!(f)?;
	}
	Ok(())
    }
}
impl Matrix{
    fn zeroes(rows: usize, columns: usize) -> Self{
	let mut k = Vec::with_capacity(rows);
	for _ in 0..rows{
	    k.push(vec![0.0; columns]);
	}
	Self{
	    data: k,
	}
    }
    pub fn random(rows: usize, columns: usize, low: f64, high: f64, gen: &mut Xorrng) -> Self{
	let mut k = Vec::with_capacity(rows);
	for _ in 0..rows{
	    let mut row = Vec::with_capacity(columns);
	    for _ in 0..columns{
		row.push(gen.rand_float(low, high));
	    }
	    k.push(row);
	}
	Self{
	    data: k,
	}
    }
    pub fn as_ket(vec: Vec<f64>) -> Self{
	Self::as_bra(vec).transpose()
    }
    pub fn as_bra(vec: Vec<f64>) -> Self{
	Self{
	    data: vec![vec],
	}
    }
    pub fn from_outer_product(ket: &Self, bra: &Self) -> Self{
	let mut k = Matrix::zeroes(ket.rows(), bra.columns());
	for i in 0..k.rows(){
	    for j in 0..k.columns(){
		k.data[i][j] = ket.data[i][0] * bra.data[0][j];
	    }
	}
	k
    }
}
impl Matrix{
    fn rows(&self) -> usize{
	self.data.len()
    }
    fn columns(&self) -> usize{
	self.data[0].len()
    }
    pub fn combine<F>(&self, other: &Self, binary_function: F) -> Matrix
    where
	F: Fn(f64, f64) -> f64
    {
	let mut k = Matrix::zeroes(self.rows(), self.columns());
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		k.data[i][j] = binary_function(self.data[i][j], other.data[i][j]);
	    }
	}
	k
    }
    pub fn combine_mut<F>(&mut self, other: &Self, binary_function: F)
    where
	F: Fn(f64, f64) -> f64
    {
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		self.data[i][j] = binary_function(self.data[i][j], other.data[i][j]);
	    }
	}
    }
    pub fn apply<F>(&self, unary_function: F) -> Matrix
    where
	F: Fn(f64) -> f64
    {
	let mut k = Matrix::zeroes(self.rows(), self.columns());
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		k.data[i][j] = unary_function(self.data[i][j]);
	    }
	}
	k
    }
    fn apply_mut<F>(&mut self, unary_function: F)
    where
	F: Fn(f64) -> f64
    {
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		self.data[i][j] = unary_function(self.data[i][j]);
	    }
	}
    }
   
    pub fn mult(&self, other: &Self) -> Self{
	let mut k = Self::zeroes(self.rows(), other.columns());
	for i in 0..self.rows(){
	    for j in 0..other.columns(){
		let mut acc = 0.0;
		for k in 0..self.columns(){
		    acc += self.data[i][k] * other.data[j][k];
		}
		k.data[i][j] = acc;
	    }
	}
	k
    }
    pub fn to_scalar<F>(&self, binary_function: F) -> f64
    where
	F: Fn(f64, f64) -> f64
    {
	let mut acc = 0.0;
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		acc = binary_function(acc, self.data[i][j]);
	    }
	}
	acc
    }
    pub fn transpose(&self) -> Self {
	let mut k = Matrix::zeroes(self.columns(), self.rows());
	for i in 0..k.rows(){
	    for j in 0..k.columns(){
		k.data[i][j] = self.data[j][i];
	    }
	}
	k
    }
}
#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn identity(){
	let mut gen = Xorrng::seed(23432);
	let j = Matrix::random(2, 2, 0.0, 1.0, &mut gen);
	let id = Matrix{
	    data: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
	};
	assert_eq!(j.mult(&id), j);
    }
    #[test]
    fn addition(){
	let mut k = Matrix{
	    data: vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]
	};
	let l = Matrix{
	    data: vec![vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]]
	};
	let r = Matrix{
	    data: vec![vec![3.0, 5.0, 7.0], vec![5.0, 7.0, 9.0]]
	};
	k.combine_mut(&l, |a, b| a+b);
	assert_eq!(k, r);
    }
    #[test]
    fn outer(){
	let k  = Matrix{
	    data: vec![vec![1.0], vec![2.0], vec![3.0]],
	};
	let l = Matrix{
	    data: vec![vec![1.0, 2.0, 3.0]],
	};
	let r = Matrix{
	    data: vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 6.0, 9.0]],
	};
	assert_eq!(r, Matrix::from_outer_product(&k, &l));
    }
}
