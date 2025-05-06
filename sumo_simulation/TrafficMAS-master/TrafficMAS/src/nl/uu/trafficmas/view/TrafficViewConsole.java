package nl.uu.trafficmas.view;

import java.util.Collection;
import java.util.Map;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.organisation.Organisation;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

public class TrafficViewConsole implements TrafficView {

	StringBuilder sb;
	
	public TrafficViewConsole(){
		sb = new StringBuilder();
	}
	
	@Override
	public void updateFromRoadNetwork(RoadNetwork rn) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateFromAgent(Agent agent) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateFromAgents(Collection<Agent> agents) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateFromOrganisation(Organisation org) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateFromOrganisations(Map<String, Organisation> organisations) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void visualize() {
		// TODO Auto-generated method stub
		if(sb.length() > 0) {
			System.out.println(sb.toString());
			sb = new StringBuilder();
		}
	}
	
	@Override
	public void addMessage(String message) {
		sb.append(message+"\r\n");
	}

	@Override
	public void initialize() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void close() {
		// TODO Auto-generated method stub
		
	}
}
