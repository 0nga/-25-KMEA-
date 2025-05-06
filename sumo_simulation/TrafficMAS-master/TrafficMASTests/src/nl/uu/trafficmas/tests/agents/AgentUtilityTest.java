package nl.uu.trafficmas.tests.agents;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;

public class AgentUtilityTest {

	@Test
	public void agentUtility() {
		AgentProfileType[] agentProfileTypes 	= AgentProfileType.values();
		ArrayList<Edge> lst 					= new ArrayList<Edge>();
		Route route = new Route("route0", lst);
		// goal achieved should be utility 1
		for(AgentProfileType apt : agentProfileTypes) {
			Agent agent = apt.toAgent(Agent.getNextAgentID(),null, route, null, 10, 10);
			// Except for sumoDefAgent
			if(!apt.equals(AgentProfileType.SUMODefault)){
				assertEquals(1.0,agent.utility(agent.getGoalArrivalTime(), null, null),0);
			}
		}
	}
}
