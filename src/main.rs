use nn::model::Model;
use nn::dataset;
fn main(){
    let mut model = Model::new(vec![784, 16, 10]);
    println!("{}", model.biases[0].rows());
    println!("{}", model.biases[1].rows());
    println!("{}", model.biases[2].rows());
    let mut d = dataset::read().unwrap();
    //println!("{:?}", dataset::translate(&d[54333]));
    for i in 0..100{
	let pair = dataset::translate(&d[i]);
	let deltas = model.do_one_example(pair.0, pair.1);
	model.update(deltas);
    }
}
