package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class MergeLaneNormScheme extends NormScheme {
	
	
	private Sensor mainSensor;
	private Set<String> tickedAgents;

	public MergeLaneNormScheme(String id, SanctionType sanctionType,
			List<Sensor> sensorList) {
		super(id, sanctionType, sensorList);
		mainSensor 	= sensorList.get(0);
		tickedAgents= new HashSet<String>();	
	}

	@Override
	public List<NormInstantiation> instantiateNorms(RoadNetwork rn,
			int currentTime, Map<String, AgentData> currentOrgKnowledge) {
		
		List<NormInstantiation> normInstList = new ArrayList<NormInstantiation>();
		
		List<AgentData> mainList = mainSensor.readSensor();
		
		removeTickedAgents(mainList);
		
		
		for (int i = 0; i < mainList.size(); i++) {
			tickedAgents.add(mainList.get(i).id);
		}
		
		for(AgentData ad : mainList){
			System.out.println(ad.id);
			MergeLaneNormInstantiation ni = new MergeLaneNormInstantiation(this, ad.id);
			ni.setLaneIndex(1);
			normInstList.add(ni);
		}
		
		return normInstList;
	}

	@Override
	public boolean checkCondition(Map<String, AgentData> currentOrgKnowledge) {
		for(Entry<String, AgentData> entry : currentOrgKnowledge.entrySet()){
			//boolean mainSensorTriggered = entry.getValue().roadID.equals(mainSensor.lane.getRoadID());
			boolean isNotTicked = !tickedAgents.contains(entry.getKey());
			return isNotTicked;
		}
		
		
		return false;
	}

	@Override
	public boolean violated(AgentData ad) {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public List<AgentData> getGoals() {
		// TODO Auto-generated method stub
		return null;
	}
	

	private void removeTickedAgents(List<AgentData> mainList) {
		Iterator<AgentData> iter = mainList.iterator();
		while (iter.hasNext()) {
		    AgentData ad = iter.next();
		    if (tickedAgents.contains(ad.id))
		        iter.remove();
		}
	}
	
}
