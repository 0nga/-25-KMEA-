import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import nl.uu.trafficmas.controller.Statistics;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.simulationmodel.SimulationModel;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.view.DummyView;
import nl.uu.trafficmas.view.TrafficView;


public class TrafficMASExperiments {
	public static void main(String[] args) throws Exception {
		if(args.length < 4) { 
			System.out.println("At least four arguments needed");
			System.exit(1);
		}
		String dir		= args[0];
		String masXML 	= args[1];
		String sumoBin 	= args[2];
		String sumoXML	= args[3];
		int numberOfRuns= Integer.parseInt(args[4]);
		Random rand = new Random();
		long seed = rand.nextLong();

		double averageGap 				= 0;
		double averageSpeedInNetwork 	= 0;
		double sanctionsIssued 			= 0;
		double changeLaneActions		= 0;
		double throughput				= 0;
		double startTimeAllExp 			= System.nanoTime();
		int crashes						= 0;
		System.out.println("Going to run "+ numberOfRuns +" experiments");
		for(int i = 0; i < numberOfRuns; i++) {
			long start_time = System.nanoTime();
			DataModel dataModel 			= new DataModelXML(dir,masXML);
			SimulationModel simModel 		= new SimulationModelTraaS(sumoBin,dir+sumoXML);
			TrafficView view 				= new DummyView();

			TrafficMASController trafficMas	= new TrafficMASController(dataModel, simModel, view,seed);
			long end_time = System.nanoTime();
			double difference = (end_time - start_time)/1e6;
			System.out.println("["+((double)(i+1)/numberOfRuns*100)+"%] Run #"+(i+1)+" with seed: " + seed);
			System.out.println("init time:" + difference + "ms");
			
			
			start_time = System.nanoTime();
			
			try {
				Statistics stats = trafficMas.run(dataModel, simModel, view);
				stats.save(dir,"Output "+(i+1));
				
				averageGap				+= stats.averageGap;
				averageSpeedInNetwork 	+= stats.averageSpeedInNetwork;
				sanctionsIssued			+= stats.sanctionsIssued;
				changeLaneActions		+= stats.changeLaneActions;
				throughput				+= stats.throughput;
				
			} catch(Exception exception) {
				exception.printStackTrace();
				i--;
				crashes++;
			}
			
		
			end_time = System.nanoTime();
			difference = (end_time - start_time)/1e6;
			System.out.println("run time:" + difference + "ms");
			
			seed = rand.nextLong();
		}
		
		averageGap 				/= numberOfRuns;
		averageSpeedInNetwork 	/= numberOfRuns;
		sanctionsIssued			/= numberOfRuns;
		changeLaneActions		/= numberOfRuns;
		throughput				/= numberOfRuns;
		Statistics aggregateStatistics = new Statistics(1,0);
		aggregateStatistics.averageGap 				= averageGap;
		aggregateStatistics.averageSpeedInNetwork 	= averageSpeedInNetwork;
		aggregateStatistics.sanctionsIssued			= sanctionsIssued;
		aggregateStatistics.changeLaneActions		= changeLaneActions;
		aggregateStatistics.throughput				= throughput;
		aggregateStatistics.crashes					= crashes;
		aggregateStatistics.simpleSave(dir, "AggregateOf"+numberOfRuns);
		double endTimeAllExperiments = System.nanoTime();
		double difference = (endTimeAllExperiments - startTimeAllExp)/1e6;
		System.out.println("run time:" + difference + "ms");
		
	}
}
