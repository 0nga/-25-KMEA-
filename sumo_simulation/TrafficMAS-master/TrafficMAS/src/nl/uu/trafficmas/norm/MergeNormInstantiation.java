package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.simulationmodel.AgentData;

public class MergeNormInstantiation extends NormInstantiation {

	private int laneIndex;
	private double speed;
	List<AgentData> goals;

	public MergeNormInstantiation(NormScheme ns, String agentID) {
		super(ns, agentID);
		this.speed = 0;
		goals = new ArrayList<AgentData>();
	}
	
	@Override
	public boolean violated(AgentData ad) {
		//TODO: some leeway, i.e. 3% deviation from the target speed?
		// Agent can only receive a violation while on sensor 3.
		MergeNormScheme mergeNormScheme = (MergeNormScheme)ns;
		boolean isOnRightMergeLaneSensor = mergeNormScheme.mergeSensor.readSensor().contains(ad);
		boolean isOnLeftMergeLaneSensor = mergeNormScheme.merge_1Sensor.readSensor().contains(ad);
		if(!(isOnRightMergeLaneSensor || isOnLeftMergeLaneSensor))
			return false;
		
		return (Math.abs(ad.velocity - speed) > 0.009 || (ad.laneIndex != laneIndex)) ;
	}

	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		
		AgentData agentData = currentOrgKnowledge.get(agentID);
		MergeNormScheme mergeNormScheme = (MergeNormScheme)ns;
		boolean isOnRightMergeLaneSensor = mergeNormScheme.mergeSensor.readSensor().contains(agentData);
		boolean isOnLeftMergeLaneSensor = mergeNormScheme.merge_1Sensor.readSensor().contains(agentData);
		return isOnRightMergeLaneSensor || isOnLeftMergeLaneSensor;
	}

	public void setLaneIndex(int laneIndex){
		this.laneIndex = laneIndex;
		goals.add(new AgentData(null, null, -1, -1, null, laneIndex, -1, -1));
	}
	
	public int getLaneIndex(){
		return laneIndex;
	}

	public void setSpeed(double speed) {
		this.speed = speed;
		goals.add(new AgentData(null, null, -1, speed, null, -1, -1, -1));
	}

	public double getSpeed() {
		return speed;
	}
	
	public List<AgentData> getGoals() {
		return goals;
	}

	public void setSpeedAndLane(double correctedLastSpeed, int laneIndex) {
		this.speed = correctedLastSpeed;
		this.laneIndex = laneIndex;
		goals.add(new AgentData(null, null, -1, correctedLastSpeed, null, laneIndex, -1, -1));
	}
}
