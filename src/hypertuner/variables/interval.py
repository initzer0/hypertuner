import math

from .variable import Variable, OptimizedException, argm_dict


class IntervalVariable(Variable):

    def __init__(self, *args, required_neighbors=1, method="bisection", **kwargs):
        self.required_neighbors = required_neighbors
        super().__init__(*args, method=method, **kwargs)

    def __len__(self):
        if self.method == "bisection":
            total = int((self.values[1] - self.values[0]) / self.values[2])
            num = int(math.ceil(math.log(total, 2)))
            confidence = num + 2 * self.required_neighbors
            return confidence
        else:
            raise ValueError(f"method {self.method!r} not found")

    def next(self, scores: dict[None, float], optimize_method=min):
        if self.method == "bisection":
            return self._compute_next_value_bi(scores, optimize_method=optimize_method)
        else:
            raise ValueError(f"method {self.method!r} not found")

    def _round(self, val):
        start, stop, step_size = self.values
        v_ = val - start
        steps = v_ / step_size

        lower = start + int(steps - 0.5) * step_size
        upper = start + (1 + int(steps - 0.5)) * step_size
        upper = min([upper, stop])
        if abs(val - lower) < abs(val - upper):
            return lower
        return upper

    def _get_interval_center(self, start, stop, step_size):
        steps = (stop - start) / step_size + 1
        middle = int(steps / 2)
        return self._round(start + middle * step_size)

    def _get_closest_bounds(self, scores: dict, best=None, n=1, optimize_method=min):
        """returns (left, right), where left and right are lists of the n closest indices from the minimum"""
        if best is None:
            best = argm_dict(scores, optimize_method)

        keys = list(sorted(set([self._round(k) for k in scores.keys()])))
        index = keys.index(best)

        left = []
        for i in range(n):
            i2 = index - 1 - i
            if i2 >= 0:
                left.append(keys[i2])
            elif len(left) == 0:
                left.append(keys[index])
                break
            else:
                break

        right = []
        for i in range(n):
            i2 = index + 1 + i
            if i2 < len(keys):
                right.append(keys[i2])
            elif len(right) == 0:
                right.append(keys[index])
                break
            else:
                break

        return left[::-1], right

    def _compute_next_value_bi(self, scores: dict[None, float], optimize_method=min):
        """[select the next parameter value to be tested]

        Args:
            scores : dict[Any, float]
                : Output the next value for this parameter given
                  the previous observations for this parameter
        Description:
            This algorithm executes a modified version of the
            Bisection Method (https://en.wikipedia.org/wiki/Bisection_method).
            It finds a discreete local optimum, with a given number of
            confidence neighbors.
        """

        # 0. check initial value
        iv = self.get_initial_value()
        if iv not in scores:
            return iv

        # 1. check boundaries
        start, stop, step_size = self.values
        if start not in scores:
            return start
        if stop not in scores:
            return stop

        # 2. calculate bounds and wanted bounds
        left_done = False
        right_done = False
        best_value = argm_dict(scores, optimize_method)
        left_bound, right_bound = self._get_closest_bounds(
            scores,
            best_value,
            n=self.required_neighbors,
            optimize_method=optimize_method,
        )
        wanted_right_bound = [
            best_value + (i + 1) * step_size for i in range(self.required_neighbors)
        ]
        wanted_right_bound = [self._round(w) for w in wanted_right_bound]
        wanted_right_bound = [b for b in wanted_right_bound if b <= stop]
        wanted_left_bound = [
            best_value - (i + 1) * step_size for i in range(self.required_neighbors)
        ][::-1]
        wanted_left_bound = [self._round(w) for w in wanted_left_bound]
        wanted_left_bound = [b for b in wanted_left_bound if b >= start]

        # 2.1 check if we are done on each side of the interval
        if len(wanted_left_bound) == 0 or wanted_left_bound == left_bound:
            left_done = True
            left_step_distance = 0
        else:
            if len(left_bound) > 0:
                left_step_distance = (best_value - left_bound[-1]) / step_size
            else:
                left_step_distance = (best_value - start) / step_size

        if len(wanted_right_bound) == 0 or wanted_right_bound == right_bound:
            right_done = True
            right_step_distance = 0
        else:
            if len(right_bound) > 0:
                right_step_distance = (right_bound[0] - best_value) / step_size
            else:
                right_step_distance = (stop - best_value) / step_size

        # 3 Select appropriate Parameter
        #  3.0 both done
        if left_done and right_done:
            raise OptimizedException()

        #  3.1 left done
        if left_done:
            # check if only neighbors need to be run
            if right_step_distance > self.required_neighbors:
                # we are far enough from the neighbors
                # select middle of interval
                if len(right_bound) > 0:
                    return self._get_interval_center(
                        best_value, right_bound[0], step_size
                    )
                else:
                    return self._get_interval_center(best_value, stop, step_size)
            elif right_bound == wanted_right_bound:
                # we are done, this case should not happen actually
                raise Exception()
            else:
                # select any of the remaining neighbors
                for v in wanted_right_bound:
                    if v not in right_bound:
                        return v
                # this cant happen
                raise Exception()

        #  3.2 right done
        elif right_done:
            # check if only neighbors need to be run
            if left_step_distance > self.required_neighbors:
                # we are far enough from the neighbors
                # select middle of interval
                if len(left_bound) > 0:
                    return self._get_interval_center(
                        left_bound[-1], best_value, step_size
                    )
                else:
                    return self._get_interval_center(start, best_value, step_size)
            elif left_bound == wanted_left_bound:
                # we are done, this case should not happen actually
                raise Exception()
            else:
                # select any of the remaining neighbors
                for v in wanted_left_bound:
                    if v not in left_bound:
                        return v
                # this cant happen
                raise Exception()

        #  3.3 neither done
        else:
            if left_step_distance >= right_step_distance:
                if len(left_bound) > 0:
                    ret = self._get_interval_center(
                        left_bound[-1], best_value, step_size
                    )
                else:
                    ret = self._get_interval_center(start, best_value, step_size)
            else:
                if len(right_bound) > 0:
                    ret = self._get_interval_center(
                        best_value, right_bound[0], step_size
                    )
                else:
                    ret = self._get_interval_center(best_value, stop, step_size)
            return ret
