package nl.uu.trafficmas.agent.actions;

public class ChangeVelocityMinus50Action extends ChangeVelocityAction {
	public ChangeVelocityMinus50Action(int priority) {
		super(priority);
		this.speedIncrease 	= -50;
	}
}
