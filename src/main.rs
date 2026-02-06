mod featureSelection;
mod knapsack;

use featureSelection::*;

fn main() {
    let dataset = load_dataset("data/dataset.txt");
    println!("Survival selection: {:?}", SURVIVAL_SELECTION);
    println!("Elitism: {} (Elite count: {})\n", ELITISM, if ELITISM { ELITE_COUNT } else { 0 });
    
    let mut ga = GA::new(dataset, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS, SURVIVAL_SELECTION);
    let start = std::time::Instant::now();
    let (best, history) = ga.run();
    let elapsed = start.elapsed();
    
    let n_selected = best.genes.iter().filter(|b| **b).count();
    println!("\nResults:");
    println!("Best RMSE: {:.6}", best.fitness);
    println!("Number of features selected: {}/{}", n_selected, best.genes.len());
    println!("GA run time: {:.2?}", elapsed);
    println!("Fitness cache size: {}", ga.fitness_cache.len());
    
    println!("\nWriting data to CSV files...");
    ga.write_fitness_history(&history);
    ga.write_entropy_history(&history);
    let elitism_str = if ELITISM { format!("_Elitism{}", ELITE_COUNT) } else { String::new() };
    println!("Data saved as fitness_history_{:?}{}.csv and entropy_history_{:?}{}.csv", 
             SURVIVAL_SELECTION, elitism_str, SURVIVAL_SELECTION, elitism_str);
}

/*
// KNAPSACK MODE - To use this, comment out the featureSelection main above 
// and uncomment this one. Also change the use statement to: use knapsack::*;
fn main() {
    let items = read_items("data/knapPI_12_500_1000_82.csv");
    let mut ga = GA::new(items, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS, SURVIVAL_SELECTION);
    
    let mode_name = match SURVIVAL_SELECTION {
        1 => "Simple GA (Generational)",
        2 => "Deterministic Crowding",
        3 => "Probabilistic Crowding",
        _ => "Unknown",
    };
    println!("Running with mode {}: {}", SURVIVAL_SELECTION, mode_name);
    
    let start = std::time::Instant::now();
    let (best, history) = ga.run();
    let elapsed = start.elapsed();
    println!("Best fitness: {}", best.fitness);
    println!("GA run time: {:.2?}", elapsed);
    ga.plot_fitness_history(history);
}
*/
