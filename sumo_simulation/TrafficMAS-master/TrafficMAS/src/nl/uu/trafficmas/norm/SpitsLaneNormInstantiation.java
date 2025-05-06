package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class SpitsLaneNormInstantiation extends NormInstantiation {

	List<AgentData> goals;

	
	public SpitsLaneNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
		goals = new ArrayList<AgentData>();
	}

	@Override
	public boolean violated(AgentData ad) {
		//TODO: some leeway, i.e. 3% deviation from the target speed?
		// Agent can only receive a violation while on sensor 3.
		boolean spitsModus = ((SpitsLaneNormScheme)this.ns).spitsModus;
		Sensor control1Sensor = ((SpitsLaneNormScheme)this.ns).control1Sensor;
		if(!control1Sensor.readSensor().contains(ad) || ad.roadID.equals(control1Sensor.lane.getRoadID()))
			return false;
		
		if(spitsModus){
			return false;
		} else {
			return (ad.laneIndex == 1);
		}		
	}
	
	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		
		AgentData agentData = currentOrgKnowledge.get(agentID);
		if(agentData == null)
			return false;
		SpitsLaneNormScheme spitsLaneNormScheme = (SpitsLaneNormScheme)ns;
		Sensor control0Sensor = spitsLaneNormScheme.control0Sensor;
		Sensor control1Sensor = spitsLaneNormScheme.control1Sensor;
		
		boolean onControl0Sensor = control0Sensor.readSensor().contains(agentData);
		boolean onControl1Sensor = control1Sensor.readSensor().contains(agentData);
		boolean onControl0SensorRoad = agentData.roadID.equals(control0Sensor.lane.getRoadID());
		boolean onControl1SensorRoad = agentData.roadID.equals(control1Sensor.lane.getRoadID());
		return onControl0Sensor || onControl1Sensor || onControl0SensorRoad || onControl1SensorRoad;
	}
	
	public List<AgentData> getGoals() {
		return ns.getGoals();
	}
}
