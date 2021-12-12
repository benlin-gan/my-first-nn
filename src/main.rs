use std::fmt;
use std::fmt::Formatter;
use std::fmt::Error;
#[derive(Debug, PartialEq)]
struct Matrix{
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
}
impl Matrix{
    fn rows(&self) -> usize{
	self.data.len()
    }
    fn columns(&self) -> usize{
	self.data[0].len()
    }
    fn add(&mut self, other: &Self){
	for i in 0..self.rows(){
	    for j in 0..self.columns(){
		self.data[i][j] += other.data[i][j];
	    }
	}
    }
    fn cross(&self, other: &Self) -> Self{
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
}
fn main() {
    
}
#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn identity(){
	let j = Matrix{
	    data: vec![vec![0.2, 0.3], vec![3.0, 4.5]],
	};
	let id = Matrix{
	    data: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
	};
	assert_eq!(j.cross(&id), j);
    }
}
