package nl.uu.trafficmas.tests.simulationModelTraaS;

import static org.junit.Assert.assertEquals;
import it.polito.appeal.traci.SumoTraciConnection;
import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;
import nl.uu.trafficmas.simulationmodel.StateData;

import org.junit.Test;

public class InitializeTest {

	@Test
	public void initialize() {
		
		String sumocfg = System.getProperty("user.dir")+"/tests/SimulationModelTraaS/Initialize/ConfigTest.xml";
		SumoTraciConnection conn = SimulationModelTraaS.initialize("sumo", sumocfg);
		boolean timeStep = false;
		StateData stateData = SimulationModelTraaS.getStateData(conn, timeStep);
		assertEquals(1000, stateData.currentTimeStep);
	}

}
