import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:

    def __init__(self):
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns:
        seller_id, seller_city, seller_state
        """

        sellers = self.data['sellers'].copy()

        sellers.drop(
            'seller_zip_code_prefix',
            axis=1,
            inplace=True
        )

        sellers.drop_duplicates(inplace=True)

        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns:
        seller_id, delay_to_carrier, wait_time
        """

        order_items = self.data['order_items'].copy()

        orders = self.data['orders'].query(
            "order_status == 'delivered'"
        ).copy()

        ship = order_items.merge(
            orders,
            on='order_id'
        )

        ship['shipping_limit_date'] = pd.to_datetime(
            ship['shipping_limit_date']
        )

        ship['order_delivered_carrier_date'] = pd.to_datetime(
            ship['order_delivered_carrier_date']
        )

        ship['order_delivered_customer_date'] = pd.to_datetime(
            ship['order_delivered_customer_date']
        )

        ship['order_purchase_timestamp'] = pd.to_datetime(
            ship['order_purchase_timestamp']
        )

        def delay_to_logistic_partner(d):
            days = np.mean(
                (
                    d.order_delivered_carrier_date
                    - d.shipping_limit_date
                ) / np.timedelta64(24, 'h')
            )

            return max(days, 0)

        def order_wait_time(d):
            days = np.mean(
                (
                    d.order_delivered_customer_date
                    - d.order_purchase_timestamp
                ) / np.timedelta64(24, 'h')
            )

            return days

        delay = (
            ship.groupby('seller_id')
            .apply(delay_to_logistic_partner)
            .reset_index()
        )

        delay.columns = [
            'seller_id',
            'delay_to_carrier'
        ]

        wait = (
            ship.groupby('seller_id')
            .apply(order_wait_time)
            .reset_index()
        )

        wait.columns = [
            'seller_id',
            'wait_time'
        ]

        return delay.merge(
            wait,
            on='seller_id'
        )

    def get_active_dates(self):
        """
        Returns:
        seller_id, date_first_sale, date_last_sale,
        months_on_olist
        """

        orders_approved = self.data['orders'][[
            'order_id',
            'order_approved_at'
        ]].dropna()

        orders_sellers = (
            orders_approved
            .merge(
                self.data['order_items'],
                on='order_id'
            )[[
                'order_id',
                'seller_id',
                'order_approved_at'
            ]]
            .drop_duplicates()
        )

        orders_sellers['order_approved_at'] = pd.to_datetime(
            orders_sellers['order_approved_at']
        )

        df = (
            orders_sellers
            .groupby('seller_id')
            .agg(
                date_first_sale=('order_approved_at', 'min'),
                date_last_sale=('order_approved_at', 'max')
            )
        )

        df['months_on_olist'] = (
            (
                df['date_last_sale']
                - df['date_first_sale']
            ) / np.timedelta64(30, 'D')
        ).round()

        return df

    def get_quantity(self):
        """
        Returns:
        seller_id, n_orders, quantity, quantity_per_order
        """

        order_items = self.data['order_items']

        n_orders = (
            order_items
            .groupby('seller_id')['order_id']
            .nunique()
            .reset_index(name='n_orders')
        )

        quantity = (
            order_items
            .groupby('seller_id')
            .size()
            .reset_index(name='quantity')
        )

        result = n_orders.merge(
            quantity,
            on='seller_id'
        )

        result['quantity_per_order'] = (
            result['quantity']
            / result['n_orders']
        )

        return result

    def get_sales(self):
        """
        Returns total product sales for delivered orders.

        Returns:
        seller_id, sales
        """

        delivered_orders = self.data['orders'].query(
            "order_status == 'delivered'"
        )[['order_id']]

        sales = (
            self.data['order_items']
            .merge(
                delivered_orders,
                on='order_id'
            )
            .groupby('seller_id')['price']
            .sum()
            .reset_index(name='sales')
        )

        return sales

    def get_review_costs(self):
        """
        Returns:
        seller_id, cost_of_reviews

        Review cost is calculated at order level first,
        then aggregated to seller level.
        """

        review_costs = self.order.get_review_score()[[
            'order_id',
            'review_cost'
        ]]

        order_sellers = (
            self.data['order_items'][[
                'order_id',
                'seller_id'
            ]]
            .drop_duplicates()
        )

        seller_review_costs = order_sellers.merge(
            review_costs,
            on='order_id'
        )

        result = (
            seller_review_costs
            .groupby('seller_id', as_index=False)
            ['review_cost']
            .sum()
        )

        result.rename(
            columns={
                'review_cost': 'cost_of_reviews'
            },
            inplace=True
        )

        return result

    def get_review_score(self):
        """
        Returns:
        seller_id, share_of_five_stars,
        share_of_one_stars, review_score
        """

        reviews = self.data['order_reviews'][[
            'order_id',
            'review_score'
        ]].copy()

        order_items = (
            self.data['order_items'][[
                'order_id',
                'seller_id'
            ]]
            .drop_duplicates()
        )

        reviews_sellers = reviews.merge(
            order_items,
            on='order_id'
        )

        result = (
            reviews_sellers
            .groupby('seller_id')
            .agg(
                share_of_five_stars=(
                    'review_score',
                    lambda x: (x == 5).mean()
                ),
                share_of_one_stars=(
                    'review_score',
                    lambda x: (x == 1).mean()
                ),
                review_score=(
                    'review_score',
                    'mean'
                )
            )
            .reset_index()
        )

        return result

    def get_training_data(self):
        """
        Returns seller-level training data.

        Includes:
        revenues
        cost_of_reviews
        profits
        """

        training_set = (
            self.get_seller_features()
            .merge(
                self.get_seller_delay_wait_time(),
                on='seller_id'
            )
            .merge(
                self.get_active_dates(),
                on='seller_id'
            )
            .merge(
                self.get_quantity(),
                on='seller_id'
            )
            .merge(
                self.get_sales(),
                on='seller_id'
            )
            .merge(
                self.get_review_costs(),
                on='seller_id',
                how='left'
            )
        )

        training_set['cost_of_reviews'] = (
            training_set['cost_of_reviews']
            .fillna(0)
        )

        # Olist takes 10% of product price
        # for delivered orders.
        training_set['sales_fee'] = (
            training_set['sales'] * 0.10
        )

        # 80 BRL subscription per seller per month.
        training_set['subscription_fee'] = (
            training_set['months_on_olist'] * 80
        )

        training_set['revenues'] = (
            training_set['sales_fee']
            + training_set['subscription_fee']
        )

        training_set['profits'] = (
            training_set['revenues']
            - training_set['cost_of_reviews']
        )

        return training_set
