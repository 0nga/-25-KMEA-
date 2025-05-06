package nl.uu.trafficmas.agent;

import nl.uu.trafficmas.organisation.BruteState;

public class AgentPhysical extends PhysicalObject implements BruteState {
	public final String agentID;
	protected double velocity;
	private static int currentAgentID=0;

	
	// TODO: Replace with actual AgentType functionality.
	public AgentPhysical(String agentID){
		this.agentID = agentID;
	}
	
	public double getVelocity() {
		return velocity;
	}
	public void setVelocity(double velocity) {
		this.velocity = velocity;
	}
	
	public static String getNextAgentID() {
		return "Agent "+ currentAgentID++;
	}
}
