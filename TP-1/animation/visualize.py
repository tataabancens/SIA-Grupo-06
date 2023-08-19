import time

import arcade
from arcade import Sprite, Scene, PhysicsEnginePlatformer, Camera, csscolor
from pyglet.math import Vec2
from GameInfo import GameInfo, load_map
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE,\
    TILE_SCALING, CHARACTER_SCALING, COIN_SCALING, AGENT_LIST_NAME,\
    TARGET_LIST_NAME, FLOOR_LIST_NAME, PLAYER_MOVEMENT_SPEED, PLAYER_JUMP_SPEED, GRAVITY, CELL_LIST_NAME


class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

        self.scene: Scene | None = None

        self.player_sprite: Sprite | None = None

        self.camera: Camera | None = None

        self.game_info: GameInfo = GameInfo()

        self.gui_camera: Camera | None = None

        self.time_elapsed = 0
        self.action_interval = 1

        # Set up the game here. Call on restart
    def setup(self):

        self.camera = Camera(self.width, self.height)
        self.gui_camera = Camera(self.width, self.height)

        self.scene = arcade.Scene()

        self.setup_floor()
        self.setup_cells()

    def setup_floor(self):
        size = game_info.size
        for x in range(size):
            for y in range(size):
                wall = arcade.Sprite("resources/images/grass.png", TILE_SCALING)
                wall.center_x = x * 64
                wall.center_y = y * 64
                self.scene.add_sprite(FLOOR_LIST_NAME, wall)

    def setup_cells(self):
        size = game_info.size
        for x in range(size):
            for y in range(size):
                if game_info.map[x][y] == 1:
                    tile = arcade.Sprite(":resources:images/tiles/bomb.png", TILE_SCALING)
                else:
                    tile = arcade.Sprite("resources/images/grass.png", TILE_SCALING)
                tile.center_x = x * 64
                tile.center_y = y * 64
                self.scene.add_sprite(CELL_LIST_NAME, tile)

        for agent in game_info.agents.values():
            target_sprite = arcade.Sprite("resources/images/target.png", 0.05)
            target_sprite.center_x = agent.target_position.x * 64
            target_sprite.center_y = agent.target_position.y * 64
            target_sprite.properties["id"] = agent.id
            self.scene.add_sprite(TARGET_LIST_NAME, target_sprite)

            agent_sprite = arcade.Sprite("resources/images/car2.png", 0.08)
            agent_sprite.center_x = agent.position.x * 64
            agent_sprite.center_y = agent.position.y * 64
            agent_sprite.properties["id"] = agent.id
            self.scene.add_sprite(AGENT_LIST_NAME, agent_sprite)
            

    def on_update(self, delta_time: float):
        self.center_camera_to_player()

        self.time_elapsed += delta_time
        if self.time_elapsed >= self.action_interval:
            self.time_elapsed = 0
            self.update_agent()

    def update_agent(self):
        agent = game_info.agents[game_info.turn]
        for sprite in self.scene.get_sprite_list(AGENT_LIST_NAME):
            if sprite.properties['id'] == agent.id:
                if agent.has_next_position():
                    next_pos = agent.next_position()
                    sprite.center_x = next_pos.x * 64
                    sprite.center_y = next_pos.y * 64

    def center_camera_to_player(self):
        screen_center_x = game_info.size * 0.5 * 64 - (self.camera.viewport_width * 0.5)
        screen_center_y = game_info.size * 0.5 * 64 - (self.camera.viewport_height * 0.5)

        self.camera.move_to(Vec2(screen_center_x, screen_center_y))

    # Render the screen
    def on_draw(self):
        self.clear()
        # Code to draw the screen goes here
        self.camera.use()

        self.scene.draw()

        # Dibujar el ID encima de cada sprite
        for sprite in self.scene.get_sprite_list(AGENT_LIST_NAME):
            arcade.draw_text(str(sprite.properties["id"]), sprite.center_x, sprite.center_y, arcade.color.BLACK, font_size=20, anchor_x="center")

        # Dibujar el ID encima de cada sprite
        for sprite in self.scene.get_sprite_list(TARGET_LIST_NAME):
            arcade.draw_text(str(sprite.properties["id"]), sprite.center_x, sprite.center_y, arcade.color.BLACK,
                             font_size=20, anchor_x="center")

        self.gui_camera.use()


def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == '__main__':
    game_info = load_map()
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
