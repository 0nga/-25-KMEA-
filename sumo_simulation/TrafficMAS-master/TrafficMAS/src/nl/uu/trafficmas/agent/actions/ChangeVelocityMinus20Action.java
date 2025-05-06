package nl.uu.trafficmas.agent.actions;

public class ChangeVelocityMinus20Action extends ChangeVelocityAction {
	public ChangeVelocityMinus20Action(int priority) {
		super(priority);
		this.speedIncrease 	= -20;
	}
}
