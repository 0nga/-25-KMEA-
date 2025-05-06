package nl.uu.trafficmas.agent.actions;

import nl.uu.trafficmas.agent.Agent;
import de.tudresden.sumo.util.SumoCommand;

public abstract class SumoAgentAction extends AgentAction {

	public SumoAgentAction(int priority) {
		super(priority);
	}
	
	public abstract SumoCommand getCommand(Agent agent);

}
