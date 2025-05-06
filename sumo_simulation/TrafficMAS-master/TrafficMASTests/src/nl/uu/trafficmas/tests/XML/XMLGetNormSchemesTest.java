package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.norm.NormScheme;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLGetNormSchemesTest {

	@Test
	public void test() throws ParserConfigurationException, SAXException, IOException {
		String dir 			= System.getProperty("user.dir")+"/tests/XML/GetNormSchemes/";
		
		String nodeXML			= "NodeTest.xml";
		String edgeXML			= "EdgeTest.xml";
		String normSchemeXML	= "NormSchemeTest.xml";
		String sensorXML		= "SensorTest.xml";
		
		Document nodeDoc 	= DataModelXML.loadDocument(dir, nodeXML);
		Document edgeDoc 	= DataModelXML.loadDocument(dir, edgeXML);
		Document normSchemeDoc 	= DataModelXML.loadDocument(dir, normSchemeXML);
		Document sensorDoc 	= DataModelXML.loadDocument(dir, sensorXML);
		
		RoadNetwork rn 				= DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc);
		Map<String,Sensor> sensors 	= DataModelXML.getSensors(rn, sensorDoc);
	
		Map<String,NormScheme> normSchemesMap = DataModelXML.getNormSchemes(sensors, normSchemeDoc);
		
		assertEquals(normSchemesMap.get("ramp-merging_1").id,"ramp-merging_1");	
		assertEquals(normSchemesMap.get("ramp-merging_1").sensorList.size(),3);	
	}
}
