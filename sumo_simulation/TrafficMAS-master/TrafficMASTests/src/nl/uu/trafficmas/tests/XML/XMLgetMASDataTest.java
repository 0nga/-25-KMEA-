package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.AgentProfileType;
import nl.uu.trafficmas.datamodel.DataModel;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.datamodel.MASData;

import org.junit.Test;
import org.xml.sax.SAXException;

public class XMLgetMASDataTest {

	@Test
	public void getMASData() throws SAXException, IOException, ParserConfigurationException {
		String dir 				= System.getProperty("user.dir")+"/tests/XML/GetMASData/";
		String masXML 			= "MASTest.xml";
		
		DataModel dataModel = new DataModelXML(dir, masXML);
		
		MASData masData = dataModel.getMASData();
		assertEquals(1000,masData.simulationLength);
		assertEquals(0.01,masData.spawnProbabilities.get("all"), 0.0);
		assertEquals(0.05,masData.routeAgentTypeSpawnDist.get("all").get(AgentProfileType.HotShot),0.0);
	}
}
