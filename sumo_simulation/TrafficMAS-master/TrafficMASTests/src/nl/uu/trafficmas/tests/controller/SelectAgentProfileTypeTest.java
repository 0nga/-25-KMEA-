package nl.uu.trafficmas.tests.controller;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.controller.TrafficMASController;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;

import org.junit.Test;
import org.xml.sax.SAXException;

public class SelectAgentProfileTypeTest {

	@Test
	public void selectAgentProfileTypeTest() throws SAXException, IOException, ParserConfigurationException {
		String dir = System.getProperty("user.dir")+"/tests/Controller/SelectAgentProfileType/";
		String masXML = "MASTest.xml";
		DataModel dataModel = new DataModelXML(dir,masXML);
		MASData masData = dataModel.getMASData();
		AgentProfileType agentType;
		double coinFlip;
		
		coinFlip = 0.10;
		agentType = TrafficMASController.selectAgentProfileType(coinFlip, masData.routeAgentTypeSpawnDist.get("route0"));
		assertEquals(AgentProfileType.HotShot,agentType);
		
		coinFlip = 0.30;
		agentType = TrafficMASController.selectAgentProfileType(coinFlip, masData.routeAgentTypeSpawnDist.get("route0"));
		assertEquals(AgentProfileType.Normal,agentType);
		
		coinFlip = 0.80;
		agentType = TrafficMASController.selectAgentProfileType(coinFlip, masData.routeAgentTypeSpawnDist.get("route0"));
		assertEquals(AgentProfileType.OldLady,agentType);
		
		
	}
}
