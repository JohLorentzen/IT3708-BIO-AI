use bitvec::prelude::*;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_linear::LinearRegression;

#[derive(Clone, Copy, Debug)]
pub enum SurvivalSelection {
    Generational,
    DeterministicCrowding,
    ProbabilisticCrowding,
}

pub const POPULATION_SIZE: usize = 100;
pub const MUTATION_RATE: f64 = 0.001;
pub const CROSSOVER_RATE: f64 = 0.85;
pub const GENERATIONS: usize = 50;
pub const RANDOM_SEED: u64 = 42;

pub const ELITISM: bool = true;
pub const ELITE_COUNT: usize = 5;
pub const SURVIVAL_SELECTION: SurvivalSelection = SurvivalSelection::ProbabilisticCrowding;


pub struct Dataset {
    pub features: Array2<f64>,
    pub target: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct Individual {
    pub genes: BitVec,
    pub fitness: f64,
}

impl Individual {
    fn get_selected_features(&self, features: &Array2<f64>) -> Option<Array2<f64>> {
        let selected_indices: Vec<usize> = self.genes
            .iter()
            .enumerate()
            .filter(|(_, bit)| **bit)
            .map(|(idx, _)| idx)
            .collect();
        
        if selected_indices.is_empty() {
            return None;
        }
        
        let n_samples = features.nrows();
        let n_features = selected_indices.len();
        let mut selected = Array2::zeros((n_samples, n_features));
        
        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            selected.column_mut(new_idx).assign(&features.column(old_idx));
        }
        
        Some(selected)
    }

    fn fitness(&self, dataset: &Dataset, cache: &mut HashMap<BitVec, f64>) -> f64 {
        if let Some(&cached_fitness) = cache.get(&self.genes) {
            return cached_fitness;
        }
        
        let selected_features = match self.get_selected_features(&dataset.features) {
            Some(features) => features,
            None => {
                let fitness = f64::INFINITY;
                cache.insert(self.genes.clone(), fitness);
                return fitness;
            }
        };
        
        let n_samples = dataset.features.nrows();
        let test_size = (n_samples as f64 * 0.2) as usize;
        let train_size = n_samples - test_size;
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(RANDOM_SEED);
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];
        
        let x_train = train_indices.iter()
            .map(|&i| selected_features.row(i).to_owned())
            .collect::<Vec<_>>();
        let x_train = ndarray::stack(ndarray::Axis(0), &x_train.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap();
        
        let y_train = train_indices.iter()
            .map(|&i| dataset.target[i])
            .collect::<Vec<_>>();
        let y_train = Array1::from(y_train);
        
        let x_test = test_indices.iter()
            .map(|&i| selected_features.row(i).to_owned())
            .collect::<Vec<_>>();
        let x_test = ndarray::stack(ndarray::Axis(0), &x_test.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap();
        
        let y_test = test_indices.iter()
            .map(|&i| dataset.target[i])
            .collect::<Vec<_>>();
        let y_test = Array1::from(y_test);
        
        let train_dataset = linfa::Dataset::new(x_train, y_train);
        let model = match LinearRegression::default().fit(&train_dataset) {
            Ok(m) => m,
            Err(_) => {
                let fitness = f64::INFINITY;
                cache.insert(self.genes.clone(), fitness);
                return fitness;
            }
        };
        
        let predictions = model.predict(&x_test);
        
        let mse: f64 = predictions.iter()
            .zip(y_test.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>() / test_indices.len() as f64;
        
        let rmse = mse.sqrt();
        
        cache.insert(self.genes.clone(), rmse);
        
        rmse
    }
}

pub struct GA {
    population: Vec<Individual>,
    dataset: Dataset,
    pub fitness_cache: HashMap<BitVec, f64>,
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    generations: usize,
    survival_selection: SurvivalSelection,
}

impl GA {
    pub fn new(dataset: Dataset, population_size: usize, mutation_rate: f64, crossover_rate: f64, generations: usize, survival_selection: SurvivalSelection) -> Self {
        let mut population = Vec::with_capacity(population_size);
        let mut rng = rand::thread_rng();
        let n_genes = dataset.features.ncols();
        for _ in 0..population_size {
            let mut genes = BitVec::repeat(false, n_genes);
            for i in 0..n_genes {
                if rng.gen_bool(0.4) {
                    genes.set(i, true);
                }
            }
            population.push(Individual { genes, fitness: 0.0 });
        }
        Self {
            population,
            dataset,
            fitness_cache: HashMap::new(),
            population_size,
            mutation_rate,
            crossover_rate,
            generations,
            survival_selection,
        }
    }

    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let mut rng = rand::thread_rng();
        if !rng.gen_bool(self.crossover_rate) {
            return (parent1.clone(), parent2.clone());
        }
        let n_features = self.dataset.features.ncols();
        let crossover_point = rng.gen_range(1..n_features);
        let mut child1_genes = parent1.genes.clone();
        let mut child2_genes = parent2.genes.clone();
        for i in 0..crossover_point {
            child1_genes.set(i, parent2.genes[i]);
            child2_genes.set(i, parent1.genes[i]);
        }
        (
            Individual { genes: child1_genes, fitness: 0.0 },
            Individual { genes: child2_genes, fitness: 0.0 },
        )
    }

    fn mutation(&self, individual: &Individual) -> Individual {
        let mut rng = rand::thread_rng();
        let mut mutated = individual.genes.clone();
        for i in 0..mutated.len() {
            if rng.gen_bool(self.mutation_rate) {
                let bit = mutated[i];
                mutated.set(i, !bit);
            }
        }
        return Individual { genes: mutated, fitness: 0.0 };
    }

    fn parent_selection(&self, population: &[Individual]) -> Individual {
        let max_f = population.iter().map(|i| i.fitness).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = population.iter().map(|i| max_f - i.fitness + 1e-10).sum();
        let mut rng = rand::thread_rng();
        let mut r = rng.gen_range(0.0..sum);
        for individual in population {
            let weight = max_f - individual.fitness + 1e-10;
            if r < weight {
                return individual.clone();
            }
            r -= weight;
        }
        population[0].clone()
    }

    fn deterministic_crowding(&self, parent: &Individual, child: &Individual) -> Individual {
        if child.fitness < parent.fitness {
            child.clone()
        } else {
            parent.clone()
        }
    }

    fn probabilistic_crowding(&self, parent: &Individual, child: &Individual) -> Individual {
        let sum = parent.fitness + child.fitness;
        let probability = if sum == 0.0 {
            0.5
        } else {
            (parent.fitness / sum).clamp(0.0, 1.0)
        };
        let mut rng = rand::thread_rng();
        if rng.gen_bool(probability) {
            child.clone()
        } else {
            parent.clone()
        }
    }

    fn hamming_distance(&self, individual1: &Individual, individual2: &Individual) -> usize {
        individual1.genes.iter().zip(individual2.genes.iter()).filter(|(a, b)| a != b).count()
    }

    fn calculate_entropy(&self) -> f64 {
        let n_genes = self.dataset.features.ncols();
        let pop_size = self.population.len() as f64;
        let mut entropy = 0.0;
        
        for bit_idx in 0..n_genes {
            let count_ones = self.population.iter()
                .filter(|ind| ind.genes[bit_idx])
                .count() as f64;
            
            let p_i = count_ones / pop_size;

            if p_i > 0.0 && p_i < 1.0 {
                entropy -= p_i * p_i.log2();
                let p_zero = 1.0 - p_i;
                entropy -= p_zero * p_zero.log2();
            }
        }
        entropy
    }

    fn create_children(&self) -> (Individual, Individual, Individual, Individual) {
        let parent1 = self.parent_selection(&self.population);
        let parent2 = self.parent_selection(&self.population);
        let (child1, child2) = self.crossover(&parent1, &parent2);
        let child1 = self.mutation(&child1);
        let child2 = self.mutation(&child2);
        (parent1, parent2, child1, child2)
    }

    fn evaluate_population(&mut self) {
        for individual in &mut self.population {
            individual.fitness = individual.fitness(&self.dataset, &mut self.fitness_cache);
        }
    }

    pub fn run(&mut self) -> (Individual, Vec<(usize, f64, f64, f64, f64)>) {
        let mut history = Vec::new();
        self.evaluate_population();

        for generation in 0..self.generations {
            let elite_individuals = if ELITISM {
                let mut sorted_pop = self.population.clone();
                sorted_pop.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
                sorted_pop.into_iter().take(ELITE_COUNT).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            
            let target_size = if ELITISM {
                self.population_size - ELITE_COUNT
            } else {
                self.population_size
            };

            let new_population = match self.survival_selection {
                SurvivalSelection::Generational => {
                    let mut offspring = Vec::with_capacity(target_size);
                    for _ in 0..(target_size / 2) {
                        let (_, _, mut child1, mut child2) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        child2.fitness = child2.fitness(&self.dataset, &mut self.fitness_cache);
                        offspring.push(child1);
                        offspring.push(child2);
                    }
                    if target_size % 2 == 1 {
                        let (_, _, mut child1, _) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        offspring.push(child1);
                    }
                    offspring
                }
                SurvivalSelection::DeterministicCrowding => {
                    let mut new_pop = Vec::with_capacity(target_size);
                    for _ in 0..(target_size / 2) {
                        let (parent1, parent2, mut child1, mut child2) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        child2.fitness = child2.fitness(&self.dataset, &mut self.fitness_cache);
                        
                        let dist_p1_c1 = self.hamming_distance(&parent1, &child1);
                        let dist_p1_c2 = self.hamming_distance(&parent1, &child2);
                        
                        let (survivor1, survivor2) = if dist_p1_c1 < dist_p1_c2 {
                            (self.deterministic_crowding(&parent1, &child1), 
                             self.deterministic_crowding(&parent2, &child2))
                        } else {
                            (self.deterministic_crowding(&parent1, &child2), 
                             self.deterministic_crowding(&parent2, &child1))
                        };
                        
                        new_pop.push(survivor1);
                        new_pop.push(survivor2);
                    }
                    if target_size % 2 == 1 {
                        let (parent1, _, mut child1, _) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        new_pop.push(self.deterministic_crowding(&parent1, &child1));
                    }
                    new_pop
                }
                SurvivalSelection::ProbabilisticCrowding => {
                    let mut new_pop = Vec::with_capacity(target_size);
                    for _ in 0..(target_size / 2) {
                        let (parent1, parent2, mut child1, mut child2) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        child2.fitness = child2.fitness(&self.dataset, &mut self.fitness_cache);
                        
                        let dist_p1_c1 = self.hamming_distance(&parent1, &child1);
                        let dist_p1_c2 = self.hamming_distance(&parent1, &child2);
                        
                        let (survivor1, survivor2) = if dist_p1_c1 < dist_p1_c2 {
                            (self.probabilistic_crowding(&parent1, &child1), 
                             self.probabilistic_crowding(&parent2, &child2))
                        } else {
                            (self.probabilistic_crowding(&parent1, &child2), 
                             self.probabilistic_crowding(&parent2, &child1))
                        };
                        
                        new_pop.push(survivor1);
                        new_pop.push(survivor2);
                    }
                    if target_size % 2 == 1 {
                        let (parent1, _, mut child1, _) = self.create_children();
                        child1.fitness = child1.fitness(&self.dataset, &mut self.fitness_cache);
                        new_pop.push(self.probabilistic_crowding(&parent1, &child1));
                    }
                    new_pop
                }
            };

            if ELITISM {
                self.population = elite_individuals.into_iter()
                    .chain(new_population.into_iter())
                    .collect();
            } else {
                self.population = new_population;
            }

            let min_f = self.population.iter().map(|i| i.fitness).fold(f64::INFINITY, f64::min);
            let max_f = self.population.iter().map(|i| i.fitness).fold(f64::NEG_INFINITY, f64::max);
            let mean_f = self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population_size as f64;
            let entropy = self.calculate_entropy();
            history.push((generation, min_f, mean_f, max_f, entropy));
        }
        let best = self.population.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap().clone();
        (best, history)
    }

    pub fn write_fitness_history(&self, history: &[(usize, f64, f64, f64, f64)]) {
        let elitism_suffix = if ELITISM { format!("_Elitism{}", ELITE_COUNT) } else { String::new() };
        let filename = format!("fitness_history_{:?}{}.csv", self.survival_selection, elitism_suffix);
        let mut wtr = csv::Writer::from_path(&filename).unwrap();
        
        wtr.write_record(&["generation", "min_fitness", "mean_fitness", "max_fitness", "entropy"]).unwrap();
        
        for (generation, min, mean, max, entropy) in history {
            wtr.write_record(&[
                generation.to_string(),
                min.to_string(),
                mean.to_string(),
                max.to_string(),
                entropy.to_string()
            ]).unwrap();
        }
        
        wtr.flush().unwrap();
    }

    pub fn write_entropy_history(&self, history: &[(usize, f64, f64, f64, f64)]) {
        let elitism_suffix = if ELITISM { format!("_Elitism{}", ELITE_COUNT) } else { String::new() };
        let filename = format!("entropy_history_{:?}{}.csv", self.survival_selection, elitism_suffix);
        let mut wtr = csv::Writer::from_path(&filename).unwrap();
        
        wtr.write_record(&["generation", "entropy"]).unwrap();
        
        for (generation, _, _, _, entropy) in history {
            wtr.write_record(&[
                generation.to_string(),
                entropy.to_string()
            ]).unwrap();
        }
        
        wtr.flush().unwrap();
    }
}

pub fn load_dataset(path: &str) -> Dataset {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();
    
    let mut data_rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record.iter()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        data_rows.push(row);
    }
    
    let n_rows = data_rows.len();
    let n_cols = data_rows[0].len();
    
    let mut features = Array2::zeros((n_rows, n_cols - 1));
    let mut target = Array1::zeros(n_rows);
    
    for (i, row) in data_rows.iter().enumerate() {
        for j in 0..n_cols - 1 {
            features[[i, j]] = row[j];
        }
        target[i] = row[n_cols - 1];
    }
    
    Dataset { features, target }
}
