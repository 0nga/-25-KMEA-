package nl.uu.trafficmas.tests.simulationModelTraaS;

import static org.junit.Assert.assertEquals;
import it.polito.appeal.traci.SumoTraciConnection;

import java.util.HashMap;
import java.util.LinkedHashMap;

import nl.uu.trafficmas.simulationmodel.SimulationModelTraaS;

import org.junit.Test;

import de.tudresden.sumo.cmd.Simulation;

public class InitializeWithOptionTest {

	@Test
	public void initializeWithOption() {
		String sumocfg = System.getProperty("user.dir")+"/tests/SimulationModelTraaS/InitializeWithOption/ConfigTest.xml";
		
		HashMap <String,String> options = new LinkedHashMap<>();
		options.put("step-length", "0.1");
		SumoTraciConnection conn = SimulationModelTraaS.initializeWithOptions(options, "sumo", sumocfg);
		try {
            int simtime = (int) conn.do_job_get(Simulation.getCurrentTime());
			conn.do_timestep();
			assertEquals(100, simtime);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
