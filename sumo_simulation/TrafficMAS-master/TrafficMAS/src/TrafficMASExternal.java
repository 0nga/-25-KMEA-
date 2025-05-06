import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.simulationmodel.SimulationModel;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.view.TrafficView;
import nl.uu.trafficmas.view.TrafficViewConsole;


public class TrafficMASExternal {
	public static void main(String[] args) throws Exception {
		if(args.length < 4) { 
			System.out.println("At least four arguments needed");
			System.exit(1);
		}
		String dir		= args[0];
		String masXML 	= args[1];
		String address 	= args[2];
		int port		= Integer.parseInt(args[3]);
		
		long seed = -1;
		if(args.length >= 5) {
			seed = Integer.parseInt(args[3]);
		}
		
		long start_time = System.nanoTime();
		
		DataModel dataModel 			= new DataModelXML(dir,masXML);
		SimulationModel simModel 		= new SimulationModelTraaS(address, port);
		TrafficView view 				= new TrafficViewConsole();
		TrafficMASController trafficMas	= new TrafficMASController(dataModel, simModel, view,seed);
		long end_time = System.nanoTime();
		double difference = (end_time - start_time)/1e6;
		System.out.println("inittime:" + difference + "ms");
		
		start_time = System.nanoTime();
		trafficMas.run(dataModel, simModel, view);
		end_time = System.nanoTime();
		difference = (end_time - start_time)/1e6;
		System.out.println("run time:" + difference + "ms");

	}
}
