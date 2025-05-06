package nl.uu.trafficmas.simulationmodel;

import java.util.HashMap;
import java.util.List;

public class StateData implements Data {
	public final HashMap<String, AgentData> agentsData;
	public final HashMap<String, EdgeData> edgesData;
	public final HashMap<String, LaneData> lanesData;
	public final HashMap<String,SensorData> sensorData;
	public final List<String> departedList;

	public final int currentTimeStep;
	
	public StateData(HashMap<String, AgentData> agentsData,
			HashMap<String, EdgeData> edgesData,
			HashMap<String, LaneData> lanesData,
			HashMap<String,SensorData> sensorData,
			int currentTimeStep, List<String> departedList) {
		this.agentsData 		= agentsData;
		this.edgesData 			= edgesData;
		this.lanesData 			= lanesData;
		this.sensorData			= sensorData;
		this.departedList		= departedList;
		this.currentTimeStep 	= currentTimeStep;
	}
}