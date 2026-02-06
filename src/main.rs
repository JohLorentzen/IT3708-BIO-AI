use bitvec::prelude::*;
use plotters::prelude::*;
use rand::Rng;
use serde::Deserialize;

const MAX_WEIGHT: f64 = 280785.0;
const PENALTY_FACTOR: f64 = 1000.0;
const POPULATION_SIZE: usize = 1000;
const MUTATION_RATE: f64 = 0.01;
const CROSSOVER_RATE: f64 = 0.85;
const GENERATIONS: usize = 100;

#[derive(Deserialize, Clone)]
struct DataPoint {
    i: f64,
    p: f64,
    w: f64,
}

#[derive(Clone, Debug)]
struct Individual {
    genes: BitVec,
    fitness: f64,
}

impl Individual {

    fn total_weight(&self, items: &[DataPoint]) -> f64 {
        self.genes
            .iter()
            .zip(items.iter())
            .filter(|(included, _)| **included)
            .map(|(_, item)| item.w)
            .sum()
    }
    
    fn total_profit(&self, items: &[DataPoint]) -> f64 {
        self.genes
            .iter()
            .zip(items.iter())
            .filter(|(included, _)| **included)
            .map(|(_, item)| item.p)
            .sum()
    }

    fn penalty(&self, items: &[DataPoint]) -> f64 {
        if self.total_weight(items) <= MAX_WEIGHT {
            return 0.0;
        }
        (self.total_weight(items) - MAX_WEIGHT) * PENALTY_FACTOR
    }

    fn fitness(&self, items: &[DataPoint]) -> f64 {
        self.total_profit(items) - self.penalty(items)
    }
}

struct GA {
    population: Vec<Individual>,
    items: Vec<DataPoint>,
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    generations: usize,
}

impl GA {
    fn new(items: Vec<DataPoint>, population_size: usize, mutation_rate: f64, crossover_rate: f64, generations: usize) -> Self {
        let mut population = Vec::with_capacity(population_size);
        let mut rng = rand::thread_rng();
        let n_genes = items.len();
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
            items,
            population_size,
            mutation_rate,
            crossover_rate,
            generations,
        }
    }

    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let mut rng = rand::thread_rng();
        if !rng.gen_bool(self.crossover_rate) {
            return (parent1.clone(), parent2.clone());
        }
        let crossover_point = rng.gen_range(1..self.items.len());
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
        let min_f = population.iter().map(|i| i.fitness).fold(f64::INFINITY, f64::min);
        let sum: f64 = population.iter().map(|i| i.fitness - min_f + 1e-10).sum();
        let mut rng = rand::thread_rng();
        let mut r = rng.gen_range(0.0..sum);
        for individual in population {
            let weight = individual.fitness - min_f + 1e-10;
            if r < weight {
                return individual.clone();
            }
            r -= weight;
        }
        population[0].clone()
    }

    fn create_children(&self) -> (Individual, Individual) {
        let parent1 = self.parent_selection(&self.population);
        let parent2 = self.parent_selection(&self.population);
        let (child1, child2) = self.crossover(&parent1, &parent2);
        let child1 = self.mutation(&child1);
        let child2 = self.mutation(&child2);
        (child1, child2)
    }

    fn evaluate_population(&mut self) {
        for individual in &mut self.population {
            individual.fitness = individual.fitness(&self.items);
        }
    }

    fn run(&mut self) -> (Individual, Vec<(usize, f64, f64, f64)>) {
        let mut history = Vec::new();
        self.evaluate_population();

        for gen in 0..self.generations {
            // Create offspring population of the same size
            let mut offspring = Vec::with_capacity(self.population_size);
            for _ in 0..(self.population_size / 2) {
                let (mut child1, mut child2) = self.create_children();
                child1.fitness = child1.fitness(&self.items);
                child2.fitness = child2.fitness(&self.items);
                offspring.push(child1);
                offspring.push(child2);
            }
            
            // Generational replacement: replace entire population with offspring
            self.population = offspring;

            let min_f = self.population.iter().map(|i| i.fitness).fold(f64::INFINITY, f64::min);
            let max_f = self.population.iter().map(|i| i.fitness).fold(f64::NEG_INFINITY, f64::max);
            let mean_f = self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population_size as f64;
            history.push((gen, min_f, mean_f, max_f));
        }
        let best = self.population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap().clone();
        (best, history)
    }

    fn plot_fitness_history(&self, history: Vec<(usize, f64, f64, f64)>) {
        let y_max = history.iter().map(|(_, _, _, max)| *max).fold(0.0_f64, f64::max);
        let y_min = history.iter().map(|(_, min, _, _)| *min).fold(f64::INFINITY, f64::min);
        let padding = (y_max - y_min).max(100.0) * 0.05;
        let y_range = (y_min - padding).max(0.0)..(y_max + padding);
        let root = BitMapBackend::new("fitness_history.png", (900, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Fitness per generation", ("sans-serif", 36))
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .margin(10)
            .build_cartesian_2d(0..self.generations, y_range.clone()).unwrap();
        chart.configure_mesh()
            .x_desc("Generation")
            .y_desc("Fitness")
            .axis_desc_style(("sans-serif", 20))
            .draw().unwrap();
        chart.draw_series(
            LineSeries::new(history.iter().map(|(x, min, _, _)| (*x, *min)), GREEN.stroke_width(2))
        ).unwrap().label("Min").legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));
        chart.draw_series(
            LineSeries::new(history.iter().map(|(x, _, mean, _)| (*x, *mean)), BLUE.stroke_width(2))
        ).unwrap().label("Mean").legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));
        chart.draw_series(
            LineSeries::new(history.iter().map(|(x, _, _, max)| (*x, *max)), RED.stroke_width(2))
        ).unwrap().label("Max").legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
        chart.configure_series_labels().border_style(BLACK).background_style(WHITE).draw().unwrap();
    }
}

fn read_items(path: &str) -> Vec<DataPoint> {
    let mut rdr = csv::Reader::from_path(path).unwrap();
    rdr.deserialize().map(|r| r.unwrap()).collect()
}

fn main() {
    let items = read_items("data/knapPI_12_500_1000_82.csv");
    let mut ga = GA::new(items, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS);
    let start = std::time::Instant::now();
    let (best, history) = ga.run();
    let elapsed = start.elapsed();
    println!("Best fitness: {}", best.fitness);
    println!("GA run time: {:.2?}", elapsed);
    ga.plot_fitness_history(history);
}