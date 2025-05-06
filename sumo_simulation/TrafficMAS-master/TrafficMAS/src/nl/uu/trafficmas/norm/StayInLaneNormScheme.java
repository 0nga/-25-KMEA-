package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class StayInLaneNormScheme extends NormScheme {

	private Sensor mainSensor;
	private Sensor mergeSensor;
	private Set<String> tickedAgents;

	
	public StayInLaneNormScheme(String id, SanctionType sanctionType,
			List<Sensor> sensorList) {
		super(id, sanctionType, sensorList);
		mainSensor 	= sensorList.get(3);
		mergeSensor	= sensorList.get(4);
		tickedAgents= new HashSet<String>();
	}

	@Override
	public List<NormInstantiation> instantiateNorms(RoadNetwork rn,
			int currentTime, Map<String, AgentData> currentOrgKnowledge) {

		List<AgentData> mainList = mainSensor.readSensor();
		
		removeTickedAgents(mainList);
		
		List<NormInstantiation> normInstList = new ArrayList<NormInstantiation>();
		
		for(AgentData ad : mainList){
			StayInLaneNormInstantiation ni = new StayInLaneNormInstantiation(this, ad.id);
			Object[] leader = {ad.leaderId, ad.leaderDistance};
			AgentData agentData = new AgentData(ad.id, leader, -1, -1, null, ad.laneIndex, -1, -1);
			ni.addGoal(agentData);
			normInstList.add(ni);
		}
		
		return normInstList;
	}

	@Override
	public boolean checkCondition(Map<String, AgentData> currentOrgKnowledge) {
		boolean condition = false;
		List<AgentData> mainSensList = mainSensor.readSensor();
		for(AgentData agentData : mainSensList){
			if (!tickedAgents.contains(agentData))
				return condition = true;
		}
		return condition;
	}
	
	private void removeTickedAgents(List<AgentData> mainList) {
		Iterator<AgentData> iter = mainList.iterator();
		while (iter.hasNext()) {
		    AgentData ad = iter.next();
		    if (tickedAgents.contains(ad.id))
		        iter.remove();
		}
	}

	@Override
	public boolean violated(AgentData ad) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public List<AgentData> getGoals() {
		return null;
	}

}
