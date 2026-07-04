from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
import torch


# Implementing a WSD rate scheduler class extending SequentialLR.
class WSDScheduler(SequentialLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=100,
        stable_step_ratio=0.8,
        decay_step_ratio=0.1,
        start_factor=0.001,
        end_factor=0.01,
    ):
        """
        Initialize the WSD Scheduler.

        Args:
            optimizer: The optimizer for which the scheduler is created.
            total_steps: Total number of training steps.
            warmup_steps: Number of warmup steps.
            stable_step_ratio: Ratio of steps to keep the learning rate stable.
            decay_step_ratio: Ratio of steps to decay the learning rate.
            start_factor: Initial factor for the learning rate during warmup.
            end_factor: Final factor for the learning rate at the end of training.
        """
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        stable_steps = int(total_steps * stable_step_ratio)
        stable_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=stable_steps,
        )

        before_decay_steps = warmup_steps + stable_steps
        decay_steps = int(total_steps * decay_step_ratio)
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=decay_steps,
        )

        before_finish_steps = before_decay_steps + decay_steps
        finish_steps = total_steps - before_finish_steps
        finish_scheduler = ConstantLR(
            optimizer,
            factor=end_factor,
            total_iters=finish_steps
        )

        super().__init__(
            optimizer,
            schedulers=[
                warmup_scheduler,
                stable_scheduler,
                decay_scheduler,
                finish_scheduler
            ],
            milestones=[
                warmup_steps,
                before_decay_steps,
                before_finish_steps
            ]
        )


# Implementing a WSD scheduler with torch.optim.lr_scheduler.SequentialLR.
def get_wsd_scheduler(
        optimizer,
        total_steps,
        warmup_steps=100,
        stable_step_ratio=0.8,
        decay_step_ratio=0.1,
        start_factor=0.001,
        end_factor=0.01,
    ):
    """
    Create a WSD scheduler with torch.optim.lr_scheduler.SequentialLR.

    Args:
        optimizer: The optimizer for which the scheduler is created.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        stable_step_ratio: Ratio of steps to keep the learning rate stable.
        decay_step_ratio: Ratio of steps to decay the learning rate.
        start_factor: Initial factor for the learning rate during warmup.
        end_factor: Final factor for the learning rate at the end of training.

    Returns:
        A custom learning rate scheduler.
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    stable_steps = int(total_steps * stable_step_ratio)
    stable_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=stable_steps,
    )

    before_decay_steps = warmup_steps + stable_steps
    decay_steps = int(total_steps * decay_step_ratio)
    decay_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=decay_steps,
    )

    before_finish_steps = before_decay_steps + decay_steps
    finish_steps = total_steps - before_finish_steps
    finish_scheduler = ConstantLR(
        optimizer,
        factor=end_factor,
        total_iters=finish_steps
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler, finish_scheduler],
        milestones=[warmup_steps, before_decay_steps, before_finish_steps],
    )

if __name__ == "__main__":
    # Example usage
    # optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.02)
    # total_steps = 10000
    # scheduler = WSDScheduler(optimizer, total_steps)
    # lr = []
    # for step in range(total_steps):
    #     lr.append(scheduler.get_last_lr()[0])
    #     scheduler.step()

    # import matplotlib.pyplot as plt
    # plt.plot(lr)
    # plt.savefig('wsd_scheduler.png')
    # plt.close()
    pass